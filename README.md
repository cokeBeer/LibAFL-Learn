# LibAFL-Learn
一个中文版本的 LibAFL 笔记，主要内容是 LibAFL 原理相关的内容，同时也附加一些 LibAFL 使用方面的 tips ，方便查阅和参考。

- [LibAFL-Learn](#libafl-learn)
  - [关于LibAFL](#关于libafl)
  - [如何导入 LibAFL 作为依赖库](#如何导入-libafl-作为依赖库)
  - [LibAFL 的基本功能](#libafl-的基本功能)
  - [InProcess Fuzz 的基本要素](#inprocess-fuzz-的基本要素)
  - [入门](#入门)
    - [Observer](#observer)
    - [Harness](#harness)
    - [Feedback](#feedback)
    - [Objective](#objective)
    - [State](#state)
    - [Monitor \& Manager](#monitor--manager)
  - [进阶](#进阶)
    - [Scheduler](#scheduler)
    - [Fuzzer](#fuzzer)
    - [Executor](#executor)
    - [Mutator](#mutator)
    - [Stage](#stage)
  - [Fuzz 流程分析](#fuzz-流程分析)


## 关于LibAFL
[LibAFL](https://github.com/AFLplusplus/LibAFL) 是一个使用 rust 编写的 fuzz 库， 核心代码参考自著名的 fuzz 工具 [AFL](https://github.com/google/AFL) 以及其社区版本 [AFL++](https://github.com/AFLplusplus/AFLplusplus) 。将 LibAFL 整合到你的项目中，可以快速获得可定制的 fuzz 能力 。
## 如何导入 LibAFL 作为依赖库
因为 LibAFL 是用 rust 开发的，所以你需要有一个 rust 项目才能导入 LibAFL (这里暂且不提 pylibafl )。如果你对 rust 完全不了解，请左转 [ Rust 语言圣经 ](https://course.rs/about-book.html) 。

我们在 cargo.toml 文件中导入 LibAFL
```toml
[dependencies]
libafl = "0.9.0"
```
这样你就可以在项目中使用 LibAFL 了。但是，LibAFL 的大版本存在许多 BUG，所以我一般会选择使用某一个修复了 BUG 的 commit 作为依赖库。 例如，你可以使用下面的语句导入一个指定 commit 的 LibAFL 作为依赖库, 这个 commit 是 0.9.0 版本之后的某一个 commit
```toml
[dependencies]
libafl = { git = "https://github.com/AFLplusplus/LibAFL.git", rev = "0727c80" }
```
## LibAFL 的基本功能
LibAFL 提供了两个基本功能 （其他功能我暂时也没有用到，欢迎补充）

- ForkServer Fuzz
- InProcess Fuzz

第一个功能，ForkServer Fuzz， 和 AFL 中的 ForkServer Mode 类似，需要使用者自己对目标程序插桩，然后交给 fuzzer 测试即可。使用方法可以参考 [forkserver_simple](https://github.com/AFLplusplus/LibAFL/blob/main/fuzzers/forkserver_simple/src/main.rs)，可以看做对 AFL 经典功能的复刻。第二个功能，InProcess Fuzz, 不要求使用 afl-gcc 等工具对程序插桩，可以自己定义一个目标函数，并且对其进行 fuzz，可以自己提供覆盖率，crash 信息，灵活度很高，也是我主要使用的一个功能。使用方法可以参考 [bady_fuzzer](https://github.com/AFLplusplus/LibAFL/blob/main/fuzzers/baby_fuzzer/src/main.rs) 。

## InProcess Fuzz 的基本要素
创建并运行一个 InProcess Fuzz 可能用到以下要素

- Observer：观察者
- Harness：被 fuzz 的函数
- Feedback：反馈
- Objective：目标
- State：状态
- Monitor & Manager：监控&管理
- Scheduler：调度器
- Fuzzer：fuzzer
- Mutator：变异器
- Stage：阶段
- Executor：执行器

我们一个一个介绍

## 入门
了解下面这些要素，你已经可以写一个实现自己目标的 fuzz 工具了。

### Observer
观察者代表对一些量的观察，比如观察一个数组，观察一个 map，观察一块共享内存。使用下面的语句可以针对一块共享内存创建一个观察者。其中 StdMapObserver 是常用的观察者类型。它的构造函数中传入了一个由共享内存对象转化来的可变切片，表示 `正在观察一块共享内存` 。之后如果需要这块共享内存的信息，可以直接通过观察者获取。
```rust
// Create an observation channel using shared memory
let observer = unsafe { StdMapObserver::new("shared_memory", shmem.as_mut_slice()) };
```

### Harness
Harness 表示被 fuzz 的函数，通常是一个闭包。例如下面这个闭包。其中 input 是 fuzzer 每次传递给 harness 的输入，代码编写者可以对这个输入进行判断，来选择返回 crash 信号 `ExitKind:Crash` 还是 返回正常退出信号 `ExitKind:Ok`。通常来说，代码编写者还要在 harness 函数里添加写入覆盖率信息的逻辑，引导 fuzzer。这里具体如何写入取决于 observer 的构建方式。如果 observer 观察的是一个数组，那么可以向数组写入信息，如果 observer 观察的是一块共享内存，那么可以向共享内存写入信息。
```rust
// The closure that we want to fuzz
let mut harness = |input: &BytesInput| {
    let mut buf = input.bytes();
    if buf.len() == 1 {
        return ExitKind::Crash;
    }
    WriteCoverage(); // 写入覆盖率信息
    ExitKind::Ok
};
```

### Feedback
Feedback 表示一种抽象的反馈信息。上面提到 observer 创建了针对一块共享内存的观察，那么 feedback 就构建在 observer 提供的观察信息基础上，抽象成对于 fuzzer 有引导价值的一种反馈信息。例如，下面的语句创建了一个 feedback 。
```rust
// Feedback to rate the interestingness of an input
let mut feedback = MaxMapFeedback::new(&observer);
```
Feedback 和 observer 的区别在于，observer 只是获取信息，但是 feedback 会记录和分析信息，并且通过信息判断一个测试用例是否是 interesting 的。

### Objective
Objective 表示一种抽象的目标条件。和 feedback 相同， objective 实现了 Feedback trait，也可以判断一个测试用例是否是 interesting 的。例如，下面的语句创建了一个 objective，它实际上就是一个 CrashFeedback ，会在 harness 函数返回 `ExitKind::Crash` 时，表示测试用例是 interesting 的。
```rust
// A feedback to choose if an input is a solution or not
let mut objective = CrashFeedback::new();
```
和 feedback 不同的是，如果一个 feedback 是 interesting 的，只表示这个测试用例有进一步 fuzz 的潜力。但是，如果一个 objective 是 interesting 的，则说明这个测试用例是我们真正需要的目标。

### State
State 是一组复合信息，包含随机数模块，corpus，solution，feedback 和 objective。其中随机数模块为 fuzzer 在运行过程中提供了随机性支持，corpus 设置了 interesting 的测试用例的保存方式，solutions 设置了 objective 的保存方式，feedback 和 objective 则是上面创建的 feedback 和 objective。下面是一个简单的例子
```rust
let mut state = StdState::new(
        // RNG
        StdRand::with_seed(current_nanos()),
        // Corpus that will be evolved, we keep it in memory for performance
        InMemoryCorpus::new(),
        // Corpus in which we store solutions (crashes in this example),
        // on disk so the user can get them after stopping the fuzzer
        OnDiskCorpus::new(PathBuf::from("./crashes")).unwrap(),
        // States of the feedbacks.
        // The feedbacks can report the data that should persist in the State.
        &mut feedback,
        // Same for objective feedbacks
        &mut objective,
    )
```
可以通过 state 加载语料
```rust
state.load_initial_inputs_forced(
    &mut fuzzer,
    &mut executor,
    &mut mgr,
    &[PathBuf::from("./seeds")],
)?;
```
可以通过 state 查看 solutions 状态
```rust
state.solutions().is_empty()
```
可以通过 state 获取随机数
```rust
state.rand_mut().below(16);
```

### Monitor & Manager
Monitor 表示对于 fuzz 过程的监控。Monitor 实际上不是必须的。使用下面的代码可以创建一个简单的 monitor 和 manager
```rust
// The Monitor trait define how the fuzzer stats are displayed to the user
let mon = SimpleMonitor::new(|s| println!("{s}"));
let mut mgr = SimpleEventManager::new(mon);
```
它会在运行时打印出统计信息

## 进阶
了解下面这些要素，你可以改进你的 fuzz 工具

### Scheduler
Scheduler 决定按何种顺序选取 corpus 来生成测试用例。例如一个常见的 scheduler 是 QueueScheduler， 它按照顺序从 corpus 中获取语料
```rust
let scheduler = QueueScheduler::new();
```
可以通过实现自己的 Scheduler 来决定选取语料的顺序

### Fuzzer
Fuzzer 是对整个 fuzz 库的封装。使用下面的语句可以创建一个简单的 fuzzer
```rust
let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);
```
可以通过 fuzzer 启动 fuzz
```rust
fuzzer.fuzz_loop()
```

### Executor
Executor 负责将测试用例发送给 harness 运行。下面是一个简单的 executor
```rust
// Create the executor for an in-process function with just one observer
    let mut executor = InProcessExecutor::new(
        &mut harness,
        tuple_list!(observer),
        &mut fuzzer,
        &mut state,
        &mut mgr,
    )
    .expect("Failed to create the Executor");
```

### Mutator
Mutator 表示如何对来自 corpus 的语料进行变异，生成测试用例。最常用的 mutator 是 `havoc_mutations()` 和 StdScheduledMutator 的组合
```rust
let mutator = StdScheduledMutator::new(havoc_mutations());
```
`havoc_muations()` 会返回一个 mutator 的元组，包含如下 mutator
```rust
/// Get the mutations that compose the Havoc mutator
#[must_use]
pub fn havoc_mutations() -> HavocMutationsType {
    tuple_list!(
        BitFlipMutator::new(),
        ByteFlipMutator::new(),
        ByteIncMutator::new(),
        ByteDecMutator::new(),
        ByteNegMutator::new(),
        ByteRandMutator::new(),
        ByteAddMutator::new(),
        WordAddMutator::new(),
        DwordAddMutator::new(),
        QwordAddMutator::new(),
        ByteInterestingMutator::new(),
        WordInterestingMutator::new(),
        DwordInterestingMutator::new(),
        BytesDeleteMutator::new(),
        BytesDeleteMutator::new(),
        BytesDeleteMutator::new(),
        BytesDeleteMutator::new(),
        BytesExpandMutator::new(),
        BytesInsertMutator::new(),
        BytesRandInsertMutator::new(),
        BytesSetMutator::new(),
        BytesRandSetMutator::new(),
        BytesCopyMutator::new(),
        BytesInsertCopyMutator::new(),
        BytesSwapMutator::new(),
        CrossoverInsertMutator::new(),
        CrossoverReplaceMutator::new(),
    )
}
```
StdScheduledMutator 是对这些 mutator 的进一步封装。当 StdScheduledMutator 的 `mutate` 方法被调用时，StdScheduledMutator 会在这些 mutator 中随机选择多次进行变异
```rust
    /// New default implementation for mutate.
    /// Implementations must forward mutate() to this method
    fn scheduled_mutate(
        &mut self,
        state: &mut S,
        input: &mut I,
        stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let mut r = MutationResult::Skipped;
        let num = self.iterations(state, input);
        for _ in 0..num {
            let idx = self.schedule(state, input);
            let outcome = self
                .mutations_mut()
                .get_and_mutate(idx, state, input, stage_idx)?;
            if outcome == MutationResult::Mutated {
                r = MutationResult::Mutated;
            }
        }
        Ok(r)
    }
```
可以通过实现自己的 mutator 来决定变异方式

### Stage
Stage 是对 mutator 的进一步封装。Stage 从 corpus 中获取语料，利用 mutator 进行变异，然后交给 executor 。

## Fuzz 流程分析
这里我们分析一下 `fuzz_one` 方法的实现，这个方法表示进行一次 fuzz。它首先使用 scheduler 获取当前的 corpus id，然后交给 stage 处理
```rust
    fn fuzz_one(
        &mut self,
        stages: &mut ST,
        executor: &mut E,
        state: &mut CS::State,
        manager: &mut EM,
    ) -> Result<CorpusId, Error> {
        // Init timer for scheduler
        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().start_timer();

        // Get the next index from the scheduler
        let idx = self.scheduler.next(state)?;

        // Mark the elapsed time for the scheduler
        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().mark_scheduler_time();

        // Mark the elapsed time for the scheduler
        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().reset_stage_index();

        // Execute all stages
        stages.perform_all(self, executor, state, manager, idx)?;

        // Init timer for manager
        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().start_timer();

        // Execute the manager
        manager.process(self, state, executor)?;

        // Mark the elapsed time for the manager
        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().mark_manager_time();

        Ok(idx)
    }
```
在 stages 里面调用 `perform_all` 方法，先使用第一个 stage 运行获取、运行测试用例，然后调用第二个 stages 元组
```rust
fn perform_all(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Head::State,
        manager: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        // Perform the current stage
        self.0
            .perform(fuzzer, executor, state, manager, corpus_idx)?;

        // Execute the remaining stages
        self.1
            .perform_all(fuzzer, executor, state, manager, corpus_idx)
    }
```
在里面继续调用到了 `perform_mutational` 方法，根据从 scheduler 获取的 corpus id，选择对应的语料，变异成测试用例，交给 executor 执行，重复随机次。
```rust
 /// Runs this (mutational) stage for the given testcase
    #[allow(clippy::cast_possible_wrap)] // more than i32 stages on 32 bit system - highly unlikely...
    fn perform_mutational(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Z::State,
        manager: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        let num = self.iterations(state, corpus_idx)?;

        start_timer!(state);
        let mut testcase = state.corpus().get(corpus_idx)?.borrow_mut();
        let Ok(input) = I::try_transform_from(&mut testcase, state, corpus_idx) else { return Ok(()); };
        drop(testcase);
        mark_feature_time!(state, PerfFeature::GetInputFromCorpus);

        for i in 0..num {
            let mut input = input.clone();

            start_timer!(state);
            self.mutator_mut().mutate(state, &mut input, i as i32)?;
            mark_feature_time!(state, PerfFeature::Mutate);

            // Time is measured directly the `evaluate_input` function
            let (untransformed, post) = input.try_transform_into(state)?;
            let (_, corpus_idx) = fuzzer.evaluate_input(state, executor, manager, untransformed)?;

            start_timer!(state);
            self.mutator_mut().post_exec(state, i as i32, corpus_idx)?;
            post.post_exec(state, i as i32, corpus_idx)?;
            mark_feature_time!(state, PerfFeature::MutatePostExec);
        }
        Ok(())
    }
```