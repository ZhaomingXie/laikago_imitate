//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    laikago_ = world_->addArticulatedSystem(resourceDir_+"/laikago/laikago.urdf");
    laikago_->setName("laikago");
    laikago_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    /// get robot data
    gcDim_ = laikago_->getGeneralizedCoordinateDim();
    gvDim_ = laikago_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);
    reference_.setZero(gcDim_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.0, 0.65, -1, 0.0, 0.65, -1, 0.0, 0.65, -1, 0.0, 0.65, -1;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(40.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.5);
    laikago_->setPdGains(jointPgain, jointDgain);
    laikago_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 39;
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.3);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// indices of links that should not make contact with ground
    footIndices_.insert(laikago_->getBodyIdx("FR_calf"));
    footIndices_.insert(laikago_->getBodyIdx("FL_calf"));
    footIndices_.insert(laikago_->getBodyIdx("RL_calf"));
    footIndices_.insert(laikago_->getBodyIdx("RR_calf"));

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(laikago_);
    }
  }

  void init() final { }

  void setEnvironmentTask(int i) final {
    speed_ = (i % 11 - 5) * 0.1;
    mode_ = i % 2;
  }

  void reset() final {
    speed_ = 0;
    mode_ = 1;
    total_reward_ = 0;
    sim_step_ = 0;
    phase_ = 0;
    phase_ = rand() % max_phase_;
    gv_init_[0] = speed_;
    getReference();
    laikago_->setState(reference_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    getReference();
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ += reference_.tail(nJoints_);
    pTarget_.tail(nJoints_) = pTarget12_;

    //laikago_->setState(reference_, gv_init_);

    laikago_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    phase_ += 1;
    phase_ %= max_phase_;
    sim_step_ += 1;

    updateObservation();
    computeReward();
    total_reward_ += rewards_.sum();

    return rewards_.sum();
  }

  void computeReward() {
    float joint_reward = 0, position_reward = 0, orientation_reward = 0;
    for (int i = 0; i < 12; i++) {
      joint_reward += std::pow(gc_[7 + i] - reference_[7 + i], 2);
    }
    position_reward += std::pow(reference_[0] - gc_[0], 2) + std::pow(reference_[1] - gc_[1], 2) + std::pow(reference_[2] - gc_[2], 2);
    orientation_reward += 2 * (std::pow(gc_[4]-reference_[4], 2) + std::pow(gc_[5]-reference_[5], 2) + std::pow(gc_[6]-reference_[6], 2));
    orientation_reward += 5 * (std::pow(gv_[3], 2) + std::pow(gv_[4], 2) + std::pow(gv_[5], 2));

    rewards_.record("position", std::exp(-position_reward));
    rewards_.record("orientation", std::exp(-orientation_reward));
    rewards_.record("joint", std::exp(-2*joint_reward));
  }

  void updateObservation() {
    laikago_->getState(gc_, gv_);

    obDouble_ << gc_[2], /// body height
        gc_[3], gc_[4], gc_[5], gc_[6], /// body orientation
        gc_.tail(12), /// joint angles
        gv_[0], gv_[1], gv_[2], gv_[3], gv_[4], gv_[5], /// body linear&angular velocity
        gv_.tail(12), /// joint velocity
        speed_, mode_,
        std::sin(phase_ * 3.1415 * 2 / max_phase_), std::cos(phase_ * 3.1415 * 2 / max_phase_); 
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_) * 0.0f;

    if (gc_[2] < 0.3)
      return true;
    if (std::abs(gc_[3]) < 0.9)
      return true;
    for(auto& contact: laikago_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
        return true;
    return false;
  }

  void getReference() {
    reference_[0] = speed_ * sim_step_ * 0.025;
    reference_[1] = 0;
    reference_[2] = 0.5;
    reference_[3] = 1.0, reference_[4] = 0.0, reference_[5] = 0.0, reference_[6] = 0.0;
    for (int i = 0; i < 4; i++) {
      reference_[7 + i * 3] = 0; reference_[8 + i * 3] = 0.65; reference_[9 + i * 3] = -1;
    }
    if (mode_ == 0) {
      if (phase_ <= max_phase_ / 2) {
        reference_[7] = 0; reference_[8] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * phase_ / max_phase_); reference_[9] = -1 - 0.7 * std::sin(2 * 3.1415 * phase_ / max_phase_);
        reference_[16] = 0; reference_[17] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * phase_ / max_phase_); reference_[18] = -1 - 0.7 * std::sin(2 * 3.1415 * phase_ / max_phase_);
      }
      else {
        reference_[10] = 0; reference_[11] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_); reference_[12] = -1 - 0.7 * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_);
        reference_[13] = 0; reference_[14] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_); reference_[15] = -1 - 0.7 * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_);
      }
    }
    else {
      if (phase_ <= max_phase_ / 2) {
        reference_[7] = 0; reference_[8] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * phase_ / max_phase_); reference_[9] = -1 - 0.7 * std::sin(2 * 3.1415 * phase_ / max_phase_);
        reference_[13] = 0; reference_[14] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * phase_ / max_phase_); reference_[15] = -1 - 0.7 * std::sin(2 * 3.1415 * phase_ / max_phase_);
      }
      else {
        reference_[10] = 0; reference_[11] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_); reference_[12] = -1 - 0.7 * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_);
        reference_[16] = 0; reference_[17] = 0.65 - 0.4 * speed_ * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_); reference_[18] = -1 - 0.7 * std::sin(2 * 3.1415 * (phase_ - max_phase_/2) / max_phase_);
      }
    }
  }


  void curriculumUpdate() {
  };

  float get_total_reward() {
    return float(total_reward_);
  }

  bool time_limit_reached() {
    return sim_step_ > max_sim_step_;
  }


 private:
  int gcDim_, gvDim_, nJoints_;
  int phase_ = 0, sim_step_ = 0, max_sim_step_ = 300;
  int max_phase_ = 30;
  int mode_ = 0;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* laikago_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, reference_;
  double terminalRewardCoeff_ = -10.;
  double speed_;
  double total_reward_ = 0;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
};
}