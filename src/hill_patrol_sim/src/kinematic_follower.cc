// Ideal kinematic standing-envelope proxy, not leg/contact dynamics.
// Required height_grid is a nonuniform Cartesian, y-major x,y,z support CSV.
// base_link is the box centre: add its rotated vertical half-extent to support.
// IMU is ideal and colocated at base_link; imu_hz is an explicit SIM parameter.
// Out-of-grid commands freeze the last supported pose, never invent zero ground.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/Util.hh>
#include <ignition/gazebo/components/Gravity.hh>
#include <ignition/gazebo/components/Collision.hh>
#include <ignition/gazebo/components/Geometry.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/gazebo/components/ParentEntity.hh>
#include <ignition/msgs/Utility.hh>
#include <ignition/msgs/imu.pb.h>
#include <ignition/msgs/int32.pb.h>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/twist.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>
#include <sdf/Box.hh>
#include <yaml-cpp/yaml.h>

namespace hill_patrol_sim
{
namespace sim = ignition::gazebo;
namespace math = ignition::math;
namespace msgs = ignition::msgs;

class KinematicFollower final : public sim::System,
    public sim::ISystemConfigure, public sim::ISystemPreUpdate
{
  struct Support { double z, dx, dy; };
  struct Limits { double x, y, yaw; };

  void ReadGrid(const std::string &path)
  {
    std::ifstream stream(path);
    std::string line;
    if (!std::getline(stream, line))
      throw std::runtime_error("cannot read height_grid header");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "x,y,z")
      throw std::runtime_error("height_grid must have x,y,z header");
    double lastY = 0;
    std::size_t column = 0;
    while (std::getline(stream, line))
    {
      if (line.empty()) continue;
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      double x, y, z;
      std::string extra;
      if (!(row >> x >> y >> z) || row >> extra ||
          !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        throw std::runtime_error("invalid height_grid row");
      if (ys.empty() || y != lastY)
      {
        if (!ys.empty() && (column != xs.size() || y <= lastY))
          throw std::runtime_error("height_grid must be Cartesian y-major");
        ys.push_back(y);
        lastY = y;
        column = 0;
      }
      if (ys.size() == 1)
      {
        if (!xs.empty() && x <= xs.back())
          throw std::runtime_error("height_grid x must strictly increase");
        xs.push_back(x);
      }
      else if (column >= xs.size() || x != xs[column])
        throw std::runtime_error("height_grid rows must share x coordinates");
      zs.push_back(z);
      ++column;
    }
    if (xs.size() < 2 || ys.size() < 2 || column != xs.size() ||
        zs.size() != xs.size()*ys.size())
      throw std::runtime_error("height_grid must contain a complete 2D grid");
  }

  bool Sample(double x, double y, Support &s) const
  {
    if (x < xs.front() || x > xs.back() || y < ys.front() || y > ys.back())
      return false;
    const auto ix = std::min(xs.size()-2, static_cast<std::size_t>(
        std::upper_bound(xs.begin(), xs.end(), x)-xs.begin()-1));
    const auto iy = std::min(ys.size()-2, static_cast<std::size_t>(
        std::upper_bound(ys.begin(), ys.end(), y)-ys.begin()-1));
    const double dx = xs[ix+1]-xs[ix], dy = ys[iy+1]-ys[iy];
    const double u = (x-xs[ix])/dx, v = (y-ys[iy])/dy;
    const auto n = xs.size();
    const double a = zs[iy*n+ix], b = zs[iy*n+ix+1];
    const double c = zs[(iy+1)*n+ix], d = zs[(iy+1)*n+ix+1];
    s = {(1-v)*((1-u)*a+u*b)+v*((1-u)*c+u*d),
         ((1-v)*(b-a)+v*(d-c))/dx, ((1-u)*(c-a)+u*(d-b))/dy};
    return true;
  }

  bool SupportedPose(double x, double y, double yaw, math::Pose3d &result) const
  {
    Support s;
    if (!Sample(x, y, s)) return false;
    // R = Rz(yaw) Ry(pitch) Rx(roll); its z axis equals the world surface
    // normal (-dz/dx,-dz/dy,1). Rotate gradient to yaw coordinates first.
    const double along = std::cos(yaw)*s.dx+std::sin(yaw)*s.dy;
    const double across = -std::sin(yaw)*s.dx+std::cos(yaw)*s.dy;
    const double pitch = -std::atan(along);
    const double roll = std::atan2(across, std::sqrt(1+along*along));
    const math::Quaterniond rotation(roll, pitch, yaw);
    const double extent = std::abs(rotation.XAxis().Z())*halfSize.X()+
        std::abs(rotation.YAxis().Z())*halfSize.Y()+
        std::abs(rotation.ZAxis().Z())*halfSize.Z();
    result = math::Pose3d(math::Vector3d(x, y, s.z+extent), rotation);
    return true;
  }

  public: void Configure(const sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &parameters,
      sim::EntityComponentManager &ecm, sim::EventManager &) override
  {
    model = sim::Model(entity);
    if (!model.Valid(ecm)) throw std::runtime_error("follower must attach to model");
    if (!parameters->HasElement("height_grid") || !parameters->HasElement("limits_path") ||
        !parameters->HasElement("imu_hz"))
      throw std::runtime_error("height_grid, limits_path and explicit SIM imu_hz required");
    ReadGrid(parameters->Get<std::string>("height_grid"));
    const auto yaml = YAML::LoadFile(parameters->Get<std::string>("limits_path"));
    for (const auto &entry : yaml)
    {
      Limits value{entry.second["x"].as<double>(), entry.second["y"].as<double>(),
                   entry.second["yaw"].as<double>()};
      if (!std::isfinite(value.x) || !std::isfinite(value.y) ||
          !std::isfinite(value.yaw) || value.x <= 0 || value.y <= 0 || value.yaw <= 0)
        throw std::runtime_error("invalid shared gait limits");
      limits.emplace(entry.first.as<int>(), value);
    }
    if (!limits.count(1)) throw std::runtime_error("mock initial Basic gait 1 is missing");
    const double hz = parameters->Get<double>("imu_hz");
    if (!std::isfinite(hz) || hz <= 0)
      throw std::runtime_error("imu_hz must be finite and positive");
    period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1/hz));
    if (period.count() <= 0) throw std::runtime_error("imu_hz period below clock resolution");
    // Source centre offset from the SAME model collision geometry, not a copied .285.
    // A model plugin is configured while the model is still being created, before
    // it is linked into its world, so world entity and Gravity resolution is
    // deferred to Initialize() on the first PreUpdate.
    const auto collision = ecm.EntityByComponents(sim::components::Collision(),
        sim::components::Name("standing_envelope_collision"),
        sim::components::ParentEntity(model.CanonicalLink(ecm)));
    const auto *geometry = ecm.Component<sim::components::Geometry>(collision);
    if (!geometry || !geometry->Data().BoxShape())
      throw std::runtime_error("expected canonical base_link standing envelope box");
    halfSize = geometry->Data().BoxShape()->Size()*0.5;
    if (halfSize.Min() <= 0 || !halfSize.IsFinite())
      throw std::runtime_error("invalid standing-envelope dimensions");
    truth = node.Advertise<msgs::Pose>("/sim/ground_truth/pose");
    imu = node.Advertise<msgs::IMU>("/imu");
    if (!node.Subscribe("/sim/m20_axis", &KinematicFollower::OnAxis, this) ||
        !node.Subscribe("/sim/m20_gait", &KinematicFollower::OnGait, this))
      throw std::runtime_error("cannot subscribe to normalized mock commands");
    poseMessage.set_name(model.Name(ecm));
    imuMessage.set_entity_name("base_link");
    for (auto *header : {poseMessage.mutable_header(), imuMessage.mutable_header()})
    {
      auto *frame = header->add_data();
      frame->set_key("frame_id");
      frame->add_value(header == poseMessage.mutable_header() ? "world" : "base_link");
      header->mutable_stamp();
    }
    // Allocate protobuf submessages once; normal PreUpdate reuses them.
    poseMessage.mutable_position(); poseMessage.mutable_orientation();
    imuMessage.mutable_orientation(); imuMessage.mutable_angular_velocity();
    imuMessage.mutable_linear_acceleration();
  }

  private: void OnAxis(const msgs::Twist &message)
  {
    std::lock_guard<std::mutex> lock(mutex);
    // Only planar normalized components are meaningful; nonfinite or nonplanar
    // packets fail closed. Finite ratios are clamped to the fixed [-1,1] contract.
    const std::array<double, 6> fields{message.linear().x(), message.linear().y(),
      message.linear().z(), message.angular().x(), message.angular().y(), message.angular().z()};
    if (!std::all_of(fields.begin(), fields.end(), [](double v){return std::isfinite(v);}) ||
        fields[2] != 0 || fields[3] != 0 || fields[4] != 0)
      axis = {0, 0, 0};
    else
      axis = {std::clamp(fields[0], -1.0, 1.0), std::clamp(fields[1], -1.0, 1.0),
              std::clamp(fields[5], -1.0, 1.0)};
  }

  private: void OnGait(const msgs::Int32 &message)
  {
    std::lock_guard<std::mutex> lock(mutex);
    gait = message.data();  // Unknown applied gait freezes motion until a known gait arrives.
  }

  // Resolves the real world through the completed parent chain and takes the
  // actual gravity vector from the ECS. Called once, from the first PreUpdate,
  // because a model plugin is configured before its world linkage exists.
  // Returns false, leaving the model uncommanded, until the world entity and
  // its Gravity component are present.
  private: bool Initialize(sim::EntityComponentManager &ecm)
  {
    const auto world = sim::worldEntity(model.Entity(), ecm);
    const auto *g = world == sim::kNullEntity ? nullptr :
        ecm.Component<sim::components::Gravity>(world);
    if (!g) return false;
    const auto start = sim::worldPose(model.Entity(), ecm);
    math::Pose3d supported;
    if (!SupportedPose(start.Pos().X(), start.Pos().Y(), start.Rot().Yaw(), supported))
      throw std::runtime_error("initial model pose outside support grid");
    gravity = g->Data();
    initial = supported;
    pose = supported;
    return true;
  }

  public: void PreUpdate(const sim::UpdateInfo &info, sim::EntityComponentManager &ecm) override
  {
    if (info.paused || info.dt.count() <= 0) return;
    if (!ready)
    {
      if (!Initialize(ecm)) return;  // no pose command, IMU or truth before ready
      ready = true;
    }
    if (!initialized || info.simTime < lastTime)
    {
      if (initialized) pose = initial;
      previousPose = pose;
      previousVelocity = math::Vector3d::Zero;
      derivativeSamples = 0;
      nextImu = info.simTime;
      initialized = true;
    }
    const double dt = std::chrono::duration<double>(info.dt).count();
    std::array<double, 3> command;
    int applied;
    {
      std::lock_guard<std::mutex> lock(mutex);
      command = axis;
      applied = gait;
    }
    const auto found = limits.find(applied);
    if (found != limits.end())
    {
      const double yaw = pose.Rot().Yaw(), c = std::cos(yaw), s = std::sin(yaw);
      const auto &max = found->second;
      const double vx = command[0]*max.x, vy = command[1]*max.y;
      math::Pose3d candidate;
      if (SupportedPose(pose.Pos().X()+(c*vx-s*vy)*dt,
                        pose.Pos().Y()+(s*vx+c*vy)*dt,
                        yaw+command[2]*max.yaw*dt, candidate)) pose = candidate;
    }
    model.SetWorldPoseCmd(ecm, pose);
    const auto velocity = (pose.Pos()-previousPose.Pos())/dt;
    // Nonuniform-dt second difference uses velocity sample midpoints.
    const auto acceleration = derivativeSamples > 1 ?
        (velocity-previousVelocity)/(0.5*(dt+previousDt)) : math::Vector3d::Zero;
    // Quaternion log of world-frame increment, expressed in current body axes.
    auto delta = pose.Rot()*previousPose.Rot().Inverse();
    delta.Normalize();
    if (delta.W() < 0) delta = math::Quaterniond(-delta.W(), -delta.X(), -delta.Y(), -delta.Z());
    const math::Vector3d vector(delta.X(), delta.Y(), delta.Z());
    const double length = vector.Length();
    // ToAxis() has a near-zero tolerance unsuitable for 1 ms IMU increments.
    // atan2 retains small-angle precision; the exact zero increment is zero.
    const auto omegaWorld = length > 0 ?
        vector*(2*std::atan2(length, delta.W())/(length*dt)) : math::Vector3d::Zero;
    const auto omega = pose.Rot().RotateVectorReverse(omegaWorld);
    const auto specificForce = pose.Rot().RotateVectorReverse(acceleration-gravity);
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
    for (auto *header : {poseMessage.mutable_header(), imuMessage.mutable_header()})
    {
      header->mutable_stamp()->set_sec(ns/1000000000);
      header->mutable_stamp()->set_nsec(ns%1000000000);
    }
    msgs::Set(poseMessage.mutable_position(), pose.Pos());
    msgs::Set(poseMessage.mutable_orientation(), pose.Rot());
    truth.Publish(poseMessage);
    if (info.simTime >= nextImu)
    {
      msgs::Set(imuMessage.mutable_orientation(), pose.Rot());
      msgs::Set(imuMessage.mutable_angular_velocity(), omega);
      msgs::Set(imuMessage.mutable_linear_acceleration(), specificForce);
      imu.Publish(imuMessage);
      // No burst of duplicate samples after a large simulation-time jump.
      nextImu += period*((info.simTime-nextImu)/period+1);
    }
    previousPose = pose;
    previousVelocity = velocity;
    previousDt = dt;
    if (derivativeSamples < 2) ++derivativeSamples;
    lastTime = info.simTime;
  }

  private: sim::Model model{sim::kNullEntity};
  private: ignition::transport::Node node;
  private: ignition::transport::Node::Publisher truth, imu;
  private: msgs::Pose poseMessage;
  private: msgs::IMU imuMessage;
  private: std::vector<double> xs, ys, zs;
  private: std::unordered_map<int, Limits> limits;
  private: std::mutex mutex;
  private: std::array<double, 3> axis{0, 0, 0};
  private: int gait{1};  // Existing mock protocol initial applied gait, not invented feedback.
  private: math::Vector3d halfSize, gravity, previousVelocity;
  private: math::Pose3d pose, initial, previousPose;
  private: std::chrono::steady_clock::duration period{}, nextImu{}, lastTime{};
  private: double previousDt{0};
  private: unsigned derivativeSamples{0};
  private: bool initialized{false};
  private: bool ready{false};
};
}  // namespace hill_patrol_sim

IGNITION_ADD_PLUGIN(hill_patrol_sim::KinematicFollower, ignition::gazebo::System,
    hill_patrol_sim::KinematicFollower::ISystemConfigure,
    hill_patrol_sim::KinematicFollower::ISystemPreUpdate)
