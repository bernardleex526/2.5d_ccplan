// Ideal kinematic standing-envelope proxy, not leg/contact dynamics.
// Parameters: height_grid (nonuniform Cartesian y-major x,y,z CSV),
// limits_path (m20_limits.yaml) and sensors_path (sensors.yaml: body size,
// mid360 mount, imu rate, follower time constants).
// Planar motion: commanded body velocity through two cascaded first-order lags,
// so horizontal acceleration is continuous. Height/roll/pitch: plane fitted to
// the support under the four footprint corners, through a ramp-tracking
// third-order filter (1+3τs)/(1+τs)^3, so vertical and angular acceleration are
// continuous and a constant slope is tracked without steady-state error.
// IMU: colocated with mid360_frame (Mid-360 built-in IMU), derived from the
// smooth trajectory by central differences one IMU period late, stamped at the
// sample time. Off-grid motion is refused; the model never leaves the support.
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
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
#include <ignition/msgs/Utility.hh>
#include <ignition/msgs/imu.pb.h>
#include <ignition/msgs/int32.pb.h>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/twist.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>
#include <yaml-cpp/yaml.h>

namespace hill_patrol_sim
{
namespace sim = ignition::gazebo;
namespace math = ignition::math;
namespace msgs = ignition::msgs;
using Duration = std::chrono::steady_clock::duration;

class KinematicFollower final : public sim::System,
    public sim::ISystemConfigure, public sim::ISystemPreUpdate
{
  struct Limits { double x, y, yaw; };
  struct Target { double z, roll, pitch; };
  struct Sample { Duration t; math::Vector3d p; math::Quaterniond q; };

  // Ramp-tracking third-order filter: y = 3 w2 - 2 w3 realises (1+3τs)/(1+τs)^3.
  struct Filter
  {
    double w1{0}, w2{0}, w3{0};
    void Reset(double v) { w1 = w2 = w3 = v; }
    void Step(double u, double a) { w1 += a*(u-w1); w2 += a*(w1-w2); w3 += a*(w2-w3); }
    double Value() const { return 3*w2 - 2*w3; }
  };

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
      if (line.empty() || line == "\r") continue;
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

  bool Height(double x, double y, double &z) const
  {
    if (x < xs.front() || x > xs.back() || y < ys.front() || y > ys.back())
      return false;
    const auto ix = std::min(xs.size()-2, static_cast<std::size_t>(
        std::upper_bound(xs.begin(), xs.end(), x)-xs.begin()-1));
    const auto iy = std::min(ys.size()-2, static_cast<std::size_t>(
        std::upper_bound(ys.begin(), ys.end(), y)-ys.begin()-1));
    const double u = (x-xs[ix])/(xs[ix+1]-xs[ix]), v = (y-ys[iy])/(ys[iy+1]-ys[iy]);
    const auto n = xs.size();
    z = (1-v)*((1-u)*zs[iy*n+ix]+u*zs[iy*n+ix+1]) + v*((1-u)*zs[(iy+1)*n+ix]+u*zs[(iy+1)*n+ix+1]);
    return true;
  }

  // Least-squares plane through the support under the four footprint corners.
  bool SupportTarget(double x, double y, double yaw, Target &target) const
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double hx = halfSize.X(), hy = halfSize.Y();
    std::array<double, 4> z{};  // front-left, front-right, rear-left, rear-right
    const std::array<std::array<double, 2>, 4> corners{{{hx, hy}, {hx, -hy}, {-hx, hy}, {-hx, -hy}}};
    for (std::size_t i = 0; i < 4; ++i)
    {
      const double bx = corners[i][0], by = corners[i][1];
      if (!Height(x + c*bx - s*by, y + s*bx + c*by, z[i])) return false;
    }
    const double centre = (z[0]+z[1]+z[2]+z[3])/4;
    const double along = ((z[0]+z[1]) - (z[2]+z[3]))/(4*hx);
    const double across = ((z[0]+z[2]) - (z[1]+z[3]))/(4*hy);
    // R = Rz(yaw) Ry(pitch) Rx(roll) with body z along the plane normal.
    target.pitch = -std::atan(along);
    target.roll = std::atan2(across, std::sqrt(1+along*along));
    const math::Quaterniond rotation(target.roll, target.pitch, yaw);
    target.z = centre + std::abs(rotation.XAxis().Z())*halfSize.X() +
        std::abs(rotation.YAxis().Z())*halfSize.Y() + std::abs(rotation.ZAxis().Z())*halfSize.Z();
    return true;
  }

  static double Positive(const YAML::Node &node, const std::string &name)
  {
    const double value = node["value"].as<double>();
    if (!std::isfinite(value) || value <= 0)
      throw std::runtime_error(name + " must be finite and positive");
    return value;
  }

  public: void Configure(const sim::Entity &entity,
      const std::shared_ptr<const sdf::Element> &parameters,
      sim::EntityComponentManager &ecm, sim::EventManager &) override
  {
    model = sim::Model(entity);
    if (!model.Valid(ecm)) throw std::runtime_error("follower must attach to model");
    for (const auto *name : {"height_grid", "limits_path", "sensors_path"})
      if (!parameters->HasElement(name))
        throw std::runtime_error(std::string("missing follower parameter ") + name);
    ReadGrid(parameters->Get<std::string>("height_grid"));
    for (const auto &entry : YAML::LoadFile(parameters->Get<std::string>("limits_path")))
    {
      Limits value{entry.second["x"].as<double>(), entry.second["y"].as<double>(),
                   entry.second["yaw"].as<double>()};
      if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.yaw) ||
          value.x <= 0 || value.y <= 0 || value.yaw <= 0)
        throw std::runtime_error("invalid gait limits");
      limits.emplace(entry.first.as<int>(), value);
    }
    const auto sensors = YAML::LoadFile(parameters->Get<std::string>("sensors_path"));
    const auto size = sensors["body"]["size"]["value"];
    halfSize = math::Vector3d(size[0].as<double>(), size[1].as<double>(), size[2].as<double>())*0.5;
    if (!halfSize.IsFinite() || halfSize.Min() <= 0)
      throw std::runtime_error("invalid body size");
    const auto mount = sensors["mid360"]["mount"]["value"];
    imuMount = math::Pose3d(mount[0].as<double>(), mount[1].as<double>(), mount[2].as<double>(),
                            mount[3].as<double>(), mount[4].as<double>(), mount[5].as<double>());
    const double hz = Positive(sensors["imu"]["update_rate"], "imu.update_rate");
    period = std::chrono::duration_cast<Duration>(std::chrono::duration<double>(1/hz));
    if (period.count() <= 0) throw std::runtime_error("imu period below clock resolution");
    tauVelocity = Positive(sensors["follower"]["velocity_time_constant_s"], "velocity_time_constant_s");
    tauHeight = Positive(sensors["follower"]["height_time_constant_s"], "height_time_constant_s");

    truth = node.Advertise<msgs::Pose>("/sim/ground_truth/pose");
    imu = node.Advertise<msgs::IMU>("/imu");
    if (!node.Subscribe("/sim/m20_axis", &KinematicFollower::OnAxis, this) ||
        !node.Subscribe("/sim/m20_gait", &KinematicFollower::OnGait, this))
      throw std::runtime_error("cannot subscribe to normalized mock commands");
    poseMessage.set_name(model.Name(ecm));
    imuMessage.set_entity_name("mid360_frame");
    for (auto *header : {poseMessage.mutable_header(), imuMessage.mutable_header()})
    {
      auto *frame = header->add_data();
      frame->set_key("frame_id");
      frame->add_value(header == poseMessage.mutable_header() ? "world" : "mid360_frame");
    }
  }

  private: void OnAxis(const msgs::Twist &message)
  {
    std::lock_guard<std::mutex> lock(mutex);
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
    gait = message.data();
  }

  // A model plugin is configured before its world linkage exists, so world
  // gravity and the starting pose are resolved on the first PreUpdate.
  private: bool Initialize(sim::EntityComponentManager &ecm)
  {
    const auto world = sim::worldEntity(model.Entity(), ecm);
    const auto *g = world == sim::kNullEntity ? nullptr :
        ecm.Component<sim::components::Gravity>(world);
    if (!g) return false;
    gravity = g->Data();
    const auto start = sim::worldPose(model.Entity(), ecm);
    initialX = start.Pos().X();
    initialY = start.Pos().Y();
    initialYaw = start.Rot().Yaw();
    Target target;
    if (!SupportTarget(initialX, initialY, initialYaw, target))
      throw std::runtime_error("initial model pose outside support grid");
    return true;
  }

  private: void Restart()
  {
    x = initialX; y = initialY; yaw = initialYaw;
    v1 = v2 = {0, 0, 0};
    Target target;
    SupportTarget(x, y, yaw, target);
    zFilter.Reset(target.z);
    rollFilter.Reset(target.roll);
    pitchFilter.Reset(target.pitch);
    samples.clear();
  }

  public: void PreUpdate(const sim::UpdateInfo &info, sim::EntityComponentManager &ecm) override
  {
    if (info.paused || info.dt.count() <= 0) return;
    if (!ready)
    {
      if (!Initialize(ecm)) return;  // no pose command, IMU or truth before ready
      ready = true;
      Restart();
      nextImu = info.simTime;
    }
    else if (info.simTime < lastTime)
    {
      Restart();
      nextImu = info.simTime;
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
    const std::array<double, 3> desired = found == limits.end() ? std::array<double, 3>{0, 0, 0} :
        std::array<double, 3>{command[0]*found->second.x, command[1]*found->second.y,
                              command[2]*found->second.yaw};
    const double av = 1 - std::exp(-dt/tauVelocity);
    for (std::size_t i = 0; i < 3; ++i)
    {
      v1[i] += av*(desired[i]-v1[i]);
      v2[i] += av*(v1[i]-v2[i]);
    }
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double nx = x + (c*v2[0]-s*v2[1])*dt, ny = y + (s*v2[0]+c*v2[1])*dt;
    const double nyaw = yaw + v2[2]*dt;
    Target target;
    if (SupportTarget(nx, ny, nyaw, target))
    {
      x = nx; y = ny; yaw = nyaw;
    }
    else
    {
      v1 = v2 = {0, 0, 0};  // Refuse to leave the support grid.
      SupportTarget(x, y, yaw, target);
    }
    const double ah = 1 - std::exp(-dt/tauHeight);
    zFilter.Step(target.z, ah);
    rollFilter.Step(target.roll, ah);
    pitchFilter.Step(target.pitch, ah);
    const math::Pose3d pose(x, y, zFilter.Value(), rollFilter.Value(), pitchFilter.Value(), yaw);
    model.SetWorldPoseCmd(ecm, pose);

    Stamp(poseMessage.mutable_header(), info.simTime);
    msgs::Set(poseMessage.mutable_position(), pose.Pos());
    msgs::Set(poseMessage.mutable_orientation(), pose.Rot());
    truth.Publish(poseMessage);

    if (info.simTime >= nextImu)
    {
      const auto imuPose = pose*imuMount;
      samples.push_back({info.simTime, imuPose.Pos(), imuPose.Rot()});
      nextImu += period*((info.simTime-nextImu)/period+1);
      if (samples.size() == 3)
      {
        PublishImu();
        samples.pop_front();
      }
    }
    lastTime = info.simTime;
  }

  private: static void Stamp(msgs::Header *header, Duration t)
  {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
    header->mutable_stamp()->set_sec(ns/1000000000);
    header->mutable_stamp()->set_nsec(ns%1000000000);
  }

  // Central differences over samples[0..2], reported for samples[1].
  private: void PublishImu()
  {
    const auto &a = samples[0], &b = samples[1], &c = samples[2];
    const double h1 = std::chrono::duration<double>(b.t-a.t).count();
    const double h2 = std::chrono::duration<double>(c.t-b.t).count();
    const auto acceleration = ((c.p-b.p)/h2 - (b.p-a.p)/h1)*(2/(h1+h2));
    auto delta = a.q.Inverse()*c.q;
    delta.Normalize();
    if (delta.W() < 0) delta = math::Quaterniond(-delta.W(), -delta.X(), -delta.Y(), -delta.Z());
    const math::Vector3d vector(delta.X(), delta.Y(), delta.Z());
    const double length = vector.Length();
    // atan2 keeps small-angle precision for millisecond increments.
    const auto omega = length > 0 ?
        vector*(2*std::atan2(length, delta.W())/(length*(h1+h2))) : math::Vector3d::Zero;
    Stamp(imuMessage.mutable_header(), b.t);
    msgs::Set(imuMessage.mutable_orientation(), b.q);
    msgs::Set(imuMessage.mutable_angular_velocity(), omega);
    msgs::Set(imuMessage.mutable_linear_acceleration(), b.q.RotateVectorReverse(acceleration-gravity));
    imu.Publish(imuMessage);
  }

  private: sim::Model model{sim::kNullEntity};
  private: ignition::transport::Node::Publisher truth, imu;
  private: msgs::Pose poseMessage;
  private: msgs::IMU imuMessage;
  private: std::vector<double> xs, ys, zs;
  private: std::unordered_map<int, Limits> limits;
  private: std::mutex mutex;
  private: std::array<double, 3> axis{0, 0, 0};
  private: int gait{1};  // Mock M20 starts in Basic gait.
  // Declared after the state its callbacks lock, so it is destroyed first.
  private: ignition::transport::Node node;
  private: math::Vector3d halfSize, gravity;
  private: math::Pose3d imuMount;
  private: double tauVelocity{0}, tauHeight{0};
  private: double initialX{0}, initialY{0}, initialYaw{0}, x{0}, y{0}, yaw{0};
  private: std::array<double, 3> v1{0, 0, 0}, v2{0, 0, 0};
  private: Filter zFilter, rollFilter, pitchFilter;
  private: std::deque<Sample> samples;
  private: Duration period{}, nextImu{}, lastTime{};
  private: bool ready{false};
};
}  // namespace hill_patrol_sim

IGNITION_ADD_PLUGIN(hill_patrol_sim::KinematicFollower, ignition::gazebo::System,
    hill_patrol_sim::KinematicFollower::ISystemConfigure,
    hill_patrol_sim::KinematicFollower::ISystemPreUpdate)
