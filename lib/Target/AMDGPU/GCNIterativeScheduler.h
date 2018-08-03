//===- GCNIterativeScheduler.h - GCN Scheduler ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_GCNITERATIVESCHEDULER_H
#define LLVM_LIB_TARGET_AMDGPU_GCNITERATIVESCHEDULER_H

#include "GCNRegPressure.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Allocator.h"
#include <limits>
#include <memory>
#include <vector>
#include "llvm/CodeGen/ScheduleDFS.h"

namespace llvm {

class MachineInstr;
class SUnit;
class raw_ostream;

class SchedDFSResult2 : public SchedDFSResult {
public:
  SchedDFSResult2(unsigned Limit) :
    SchedDFSResult(/*BottomU*/true, Limit) {}

  bool isInTreeOrDescendant(const SUnit *Node, unsigned SubTreeID) const;

  bool isParentTree(unsigned PotentailParentSubTreeID,
                    unsigned PotentialChildSubTreeID) const;

  unsigned getTopMostParentSubTreeID(const SUnit *Node) const;
  unsigned getTopMostParentSubTreeID(unsigned ID) const;

  friend void writeSubtreeGraph(const SchedDFSResult2 &R, StringRef Name);
};

class GCNIterativeScheduler : public ScheduleDAGMILive {
  using BaseClass = ScheduleDAGMILive;

public:
  enum StrategyKind {
    SCHEDULE_MINREGONLY,
    SCHEDULE_MINREGFORCED,
    SCHEDULE_LEGACYMAXOCCUPANCY,
    SCHEDULE_ILP
  };

  GCNIterativeScheduler(MachineSchedContext *C,
                        StrategyKind S);

  void schedule() override;

  void startBlock(MachineBasicBlock *BB) override;
  void finishBlock() override;

  void enterRegion(MachineBasicBlock *BB,
                   MachineBasicBlock::iterator Begin,
                   MachineBasicBlock::iterator End,
                   unsigned RegionInstrs) override;

  void exitRegion() override;

  void finalizeSchedule() override;

  void computeDFSResult();

  const SchedDFSResult2* getDFSResult() const { 
    return static_cast<SchedDFSResult2*>(DFSResult);
  }

  auto getTopo() const -> const decltype(Topo)& { return Topo; }

protected:
  using ScheduleRef = ArrayRef<const SUnit *>;

  struct TentativeSchedule {
    std::vector<MachineInstr *> Schedule;
    GCNRegPressure MaxPressure;
  };

  struct Region {
    // Fields except for BestSchedule are supposed to reflect current IR state
    // `const` fields are to emphasize they shouldn't change for any schedule.
    MachineBasicBlock::iterator Begin;
    // End is either a boundary instruction or end of basic block
    const MachineBasicBlock::iterator End;
    const unsigned NumRegionInstrs;
    GCNRegPressure MaxPressure;

    // best schedule for the region so far (not scheduled yet)
    std::unique_ptr<TentativeSchedule> BestSchedule;

    std::string getName(const LiveIntervals *LIS) const;

    const MachineBasicBlock *getBB() const { return Begin->getParent(); }

    MachineBasicBlock::iterator getLastMI() const {
      return End == getBB()->end() ? std::prev(End) : End;
    }
  };

  SpecificBumpPtrAllocator<Region> Alloc;
  std::vector<Region*> Regions;

  MachineSchedContext *Context;
  const StrategyKind Strategy;
  mutable GCNUpwardRPTracker UPTracker;
  DenseMap<const MachineInstr*, MachineInstr*> M0Map;

  class BuildDAG;
  class OverrideLegacyStrategy;

  GCNRPTracker::LiveRegSet getRegionLiveIns(const Region &R) const {
    return getLiveRegsBefore(*R.Begin, *LIS);
  }

  GCNRPTracker::LiveRegSet getRegionLiveOuts(const Region &R) const {
    return getLiveRegsAfter(*std::prev(R.End), *LIS);
  }

  GCNRPTracker::LiveRegSet getRegionLiveThrough(const Region &R) const {
    return getLiveThroughRegs(*R.Begin, *std::prev(R.End), *LIS, MRI);
  }

  template <typename Range>
  GCNRegPressure getSchedulePressure(const Region &R,
                                     Range &&Schedule) const;

  GCNRegPressure getRegionPressure(MachineBasicBlock::iterator Begin,
                                   MachineBasicBlock::iterator End) const;

  GCNRegPressure getRegionPressure(const Region &R) const {
    return getRegionPressure(R.Begin, R.End);
  }

  void setBestSchedule(Region &R,
                       ScheduleRef Schedule,
                       const GCNRegPressure &MaxRP = GCNRegPressure());

  void scheduleBest(Region &R);

  std::vector<MachineInstr*> detachSchedule(ScheduleRef Schedule) const;

  void sortRegionsByPressure(unsigned TargetOcc);

  template <typename Range>
  void scheduleRegion(Region &R, Range &&Schedule,
                      const GCNRegPressure &MaxRP = GCNRegPressure());

  unsigned tryMaximizeOccupancy(unsigned TargetOcc =
                                std::numeric_limits<unsigned>::max());

  void scheduleLegacyMaxOccupancy(bool TryMaximizeOccupancy = true);
  void scheduleMinReg(bool force = false);
  void scheduleILP(bool TryMaximizeOccupancy = true);

  void removeM0Defs(MachineBasicBlock &BB);
  void restoreM0Defs(MachineBasicBlock &BB);
  void clearM0Map();

  void printRegions(raw_ostream &OS) const;
  void printSchedResult(raw_ostream &OS,
                        const Region *R,
                        const GCNRegPressure &RPBefore) const;
  void printSchedRP(raw_ostream &OS,
                    const GCNRegPressure &Before,
                    const GCNRegPressure &After) const;

  template <typename Range>
  bool validateSchedule(const Region &R, const Range &Schedule);

  void writeGraph(StringRef Name);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_GCNITERATIVESCHEDULER_H
