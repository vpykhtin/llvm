//===--  PromotePointerKernargsToGlobal.cpp - Promote Pointers To Global --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares and defines a pass which promotes pointer formal arguments
// to a kernel (i.e. pfe trampoline or HIP __global__ function) from the generic
// address space to the global address space. This transformation is valid due
// to the invariants established by both HC and HIP in accordance with an
// address passed to a kernel can only reside in the global address space. It is
// preferable to execute SelectAcceleratorCode before, as this reduces the
// workload by pruning functions that are not reachable by an accelerator.
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <utility>

using namespace llvm;
using namespace std;

namespace {
class PromotePointerKernArgsToGlobal : public ModulePass {
    // TODO: query the address space robustly.
    static constexpr unsigned int GlobalAddrSpace{1u};

    static
    pair<SmallVector<Type *, 8>, bool> makeNewArgList(const Function &F)
    {
        bool MustPromote = false;
        SmallVector<Type *, 8> NewArgs;
        for_each(F.arg_begin(), F.arg_end(), [&](const Argument &Arg) {
            if (!Arg.getType()->isPointerTy()) {
                NewArgs.push_back(Arg.getType());

                return;
            }

            NewArgs.push_back(cast<PointerType>(Arg.getType())
                ->getElementType()->getPointerTo(GlobalAddrSpace));

            MustPromote = true;
        });

        return {NewArgs, MustPromote};
    }

    static
    Function *makeNewFunction(
        Module &M, Function &F, const SmallVector<Type *, 8>& NewArgList)
    {
        Type *RetTy = F.getFunctionType()->getReturnType();
        FunctionType *NewFTy = FunctionType::get(
            RetTy, NewArgList, F.getFunctionType()->isVarArg());

        Function *NewF = Function::Create(NewFTy, F.getLinkage(), F.getName());
        NewF->copyAttributesFrom(&F);

        NewF->setSubprogram(F.getSubprogram());
        F.setSubprogram(nullptr);

        SmallVector<AttributeSet, 8> ArgAttribs;
        decltype(NewArgList.size()) ArgNo = 0;
        do {
            ArgAttribs.push_back(F.getAttributes().getParamAttributes(ArgNo));
            ++ArgNo;
        } while (ArgNo != NewArgList.size());

        NewF->setAttributes(AttributeList::get(
            F.getContext(),
            F.getAttributes().getFnAttributes(),
            F.getAttributes().getRetAttributes(),
            ArgAttribs));
        M.getFunctionList().insert(F.getIterator(), NewF);
        NewF->takeName(&F);
        NewF->getBasicBlockList().splice(NewF->begin(), F.getBasicBlockList());

        return NewF;
    }

    static
    void replaceWithPromotedFunction(Function &OldF, Function &PromotedF)
    {
        auto It0 = OldF.arg_begin();
        auto It1 = PromotedF.arg_begin();

        while (It0 != OldF.arg_end()) {
            It1->takeName(&*It0);
            if (It0->getType() == It1->getType()) {
                It0->replaceAllUsesWith(&*It1);
            }
            while (!It0->use_empty()) {
                It0->user_back()->replaceUsesOfWith(It0, It1);
            }

            ++It0;
            ++It1;
        }

        OldF.dropAllReferences();
        OldF.replaceAllUsesWith(UndefValue::get(OldF.getType()));
    }
public:
    static char ID;
    PromotePointerKernArgsToGlobal() : ModulePass{ID} {}

    bool runOnModule(Module &M) override
    {
        SmallVector<Function *, 8> OldFns;
        bool Modified = false;
        for_each(M.begin(), M.end(), [&](Function &F) {
            if (F.getCallingConv() != CallingConv::AMDGPU_KERNEL) return;

            auto NewArgList = makeNewArgList(F);

            if (!NewArgList.second) return;

            OldFns.push_back(&F);
            auto PromotedF = makeNewFunction(M, F, NewArgList.first);
            replaceWithPromotedFunction(F, *PromotedF);

            Modified = true;
        });

        for_each(OldFns.begin(), OldFns.end(), [](Function *F) {
            F->eraseFromParent();
        });

        return Modified;
    }
};
char PromotePointerKernArgsToGlobal::ID = 0;

static RegisterPass<PromotePointerKernArgsToGlobal> X{
    "promote-pointer-kernargs-to-global",
    "Promotes kernel formals of pointer type to point to the global address "
    "space, since the actuals can only represent a global address.",
    false,
    false};
}