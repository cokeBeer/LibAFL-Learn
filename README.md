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
  - [Mutators 介绍](#mutators-介绍)
    - [BitFlipMutator](#bitflipmutator)
    - [ByteFlipMutator](#byteflipmutator)
    - [ByteIncMutator](#byteincmutator)
    - [BytesDecMutator](#bytesdecmutator)
    - [ByteNegMutator](#bytenegmutator)
    - [ByteRandMutator](#byterandmutator)
    - [Byte/Word/Dword/Qword AddMutator](#byteworddwordqword-addmutator)
    - [Byte/Word/Dword InterestingMutator](#byteworddword-interestingmutator)
    - [BytesDeleteMutator](#bytesdeletemutator)
    - [BytesExpandMutator](#bytesexpandmutator)
    - [BytesInsertMutator](#bytesinsertmutator)
    - [BytesRandInsertMutator](#bytesrandinsertmutator)
    - [BytesSetMutator](#bytessetmutator)
    - [BytesRandSetMutator](#bytesrandsetmutator)
    - [BytesCopyMutator](#bytescopymutator)
    - [BytesInsertCopyMutator](#bytesinsertcopymutator)
    - [BytesSwapMutator](#bytesswapmutator)
    - [CrossoverInsertMutator](#crossoverinsertmutator)
    - [CrossoverReplaceMutator](#crossoverreplacemutator)


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
这里我们分析一下 `fuzz_one` 方法的实现，这个方法表示进行一次 fuzz。它首先使用 scheduler 的 `next` 方法获取当前的 corpus id，然后交给 stage 处理
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
`perform` 方法如下 
```rust
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut Z::State,
        manager: &mut EM,
        corpus_idx: CorpusId,
    ) -> Result<(), Error> {
        let ret = self.perform_mutational(fuzzer, executor, state, manager, corpus_idx);

        #[cfg(feature = "introspection")]
        state.introspection_monitor_mut().finish_stage();

        ret
    }
```
在里面继续调用到了 `perform_mutational` 方法，这里根据 corpus_idx 从 state 中获取指定的 testcase
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
里面调用 `evaluate_input` 
```rust
    fn evaluate_input(
        &mut self,
        state: &mut Self::State,
        executor: &mut E,
        manager: &mut EM,
        input: <Self::State as UsesInput>::Input,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input_events(state, executor, manager, input, true)
    }
```
再调用 `evaluate_input_events`
```rust
    fn evaluate_input_events(
        &mut self,
        state: &mut CS::State,
        executor: &mut E,
        manager: &mut EM,
        input: <CS::State as UsesInput>::Input,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error> {
        self.evaluate_input_with_observers(state, executor, manager, input, send_events)
    }
```
再调用 `evaluate_input_with_observers`，这里分成两段，首先调用 `execute_input` 把 testcase 喂给闭包函数，然后调用 `process_execution` 获取反馈信息
```rust
    fn evaluate_input_with_observers<E, EM>(
        &mut self,
        state: &mut Self::State,
        executor: &mut E,
        manager: &mut EM,
        input: <Self::State as UsesInput>::Input,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error>
    where
        E: Executor<EM, Self> + HasObservers<Observers = OT, State = Self::State>,
        EM: EventFirer<State = Self::State>,
    {
        let exit_kind = self.execute_input(state, executor, manager, &input)?;
        let observers = executor.observers();
        self.process_execution(state, manager, input, observers, &exit_kind, send_events)
    }
```
第一段，`execute_input` 里面会调用 `run_target`
```rust
    /// Runs the input and triggers observers and feedback
    pub fn execute_input<E, EM>(
        &mut self,
        state: &mut CS::State,
        executor: &mut E,
        event_mgr: &mut EM,
        input: &<CS::State as UsesInput>::Input,
    ) -> Result<ExitKind, Error>
    where
        E: Executor<EM, Self> + HasObservers<Observers = OT, State = CS::State>,
        EM: UsesState<State = CS::State>,
        OT: ObserversTuple<CS::State>,
    {
        start_timer!(state);
        executor.observers_mut().pre_exec_all(state, input)?;
        mark_feature_time!(state, PerfFeature::PreExecObservers);

        *state.executions_mut() += 1;

        start_timer!(state);
        let exit_kind = executor.run_target(self, state, event_mgr, input)?;
        mark_feature_time!(state, PerfFeature::TargetExecution);

        start_timer!(state);
        executor
            .observers_mut()
            .post_exec_all(state, input, &exit_kind)?;
        mark_feature_time!(state, PerfFeature::PostExecObservers);

        Ok(exit_kind)
    }
```
`run_target` 里实际就是调用之前的闭包函数 `harness_fn`，执行我们自定义的逻辑
```rust
    fn run_target(
        &mut self,
        fuzzer: &mut Z,
        state: &mut Self::State,
        mgr: &mut EM,
        input: &Self::Input,
    ) -> Result<ExitKind, Error> {
        self.handlers
            .pre_run_target(self, fuzzer, state, mgr, input);

        let ret = (self.harness_fn.borrow_mut())(input);

        self.handlers.post_run_target();
        Ok(ret)
    }
```
第二段，`process_execution` 获取反馈信息，可以看到这里通过 objective 判断当前的 testcase 是否是 solution，通过 feedback 判断当前的 testcase 是否是 interesting 的，最后在 match 块内根据结果进行相应操作
```rust
    /// Evaluate if a set of observation channels has an interesting state
    fn process_execution<EM>(
        &mut self,
        state: &mut CS::State,
        manager: &mut EM,
        input: <CS::State as UsesInput>::Input,
        observers: &OT,
        exit_kind: &ExitKind,
        send_events: bool,
    ) -> Result<(ExecuteInputResult, Option<CorpusId>), Error>
    where
        EM: EventFirer<State = Self::State>,
    {
        let mut res = ExecuteInputResult::None;

        #[cfg(not(feature = "introspection"))]
        let is_solution = self
            .objective_mut()
            .is_interesting(state, manager, &input, observers, exit_kind)?;

        #[cfg(feature = "introspection")]
        let is_solution = self
            .objective_mut()
            .is_interesting_introspection(state, manager, &input, observers, exit_kind)?;

        if is_solution {
            res = ExecuteInputResult::Solution;
        } else {
            #[cfg(not(feature = "introspection"))]
            let is_corpus = self
                .feedback_mut()
                .is_interesting(state, manager, &input, observers, exit_kind)?;

            #[cfg(feature = "introspection")]
            let is_corpus = self
                .feedback_mut()
                .is_interesting_introspection(state, manager, &input, observers, exit_kind)?;

            if is_corpus {
                res = ExecuteInputResult::Corpus;
            }
        }

        match res {
            ExecuteInputResult::None => {
                self.feedback_mut().discard_metadata(state, &input)?;
                self.objective_mut().discard_metadata(state, &input)?;
                Ok((res, None))
            }
            ExecuteInputResult::Corpus => {
                // Not a solution
                self.objective_mut().discard_metadata(state, &input)?;

                // Add the input to the main corpus
                let mut testcase = Testcase::with_executions(input.clone(), *state.executions());
                self.feedback_mut()
                    .append_metadata(state, observers, &mut testcase)?;
                let idx = state.corpus_mut().add(testcase)?;
                self.scheduler_mut().on_add(state, idx)?;

                if send_events {
                    // TODO set None for fast targets
                    let observers_buf = if manager.configuration() == EventConfig::AlwaysUnique {
                        None
                    } else {
                        Some(manager.serialize_observers::<OT>(observers)?)
                    };
                    manager.fire(
                        state,
                        Event::NewTestcase {
                            input,
                            observers_buf,
                            exit_kind: *exit_kind,
                            corpus_size: state.corpus().count(),
                            client_config: manager.configuration(),
                            time: current_time(),
                            executions: *state.executions(),
                        },
                    )?;
                }
                Ok((res, Some(idx)))
            }
            ExecuteInputResult::Solution => {
                // Not interesting
                self.feedback_mut().discard_metadata(state, &input)?;

                // The input is a solution, add it to the respective corpus
                let mut testcase = Testcase::with_executions(input, *state.executions());
                self.objective_mut()
                    .append_metadata(state, observers, &mut testcase)?;
                state.solutions_mut().add(testcase)?;

                if send_events {
                    manager.fire(
                        state,
                        Event::Objective {
                            objective_size: state.solutions().count(),
                        },
                    )?;
                }

                Ok((res, None))
            }
        }
    }
```
至此， fuzzer 完成了调度 testcase，执行 testcase，判断 testcase 是否是 solution 或者是 interesting，添加新的 corpus 和 crashes 的闭环。

## Mutators 介绍
LibAFL 内置了一些实现好的 mutator。在上面的 mutator 一节我们也列举出来了。下面介绍一下每个 mutator 具体的变异方式。
### BitFlipMutator
从 input 中随机选出一个字节，再随机选出一个比特，异或
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        let bit = 1 << state.rand_mut().choose(0..8);
        let byte = state.rand_mut().choose(input.bytes_mut());
        *byte ^= bit;
        Ok(MutationResult::Mutated)
    }
}
```

### ByteFlipMutator
从 input 中随机选出一个 byte，按位取反
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        *state.rand_mut().choose(input.bytes_mut()) ^= 0xff;
        Ok(MutationResult::Mutated)
    }
}
```

### ByteIncMutator
从 input 中随机选出一个字节，加一
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        let byte = state.rand_mut().choose(input.bytes_mut());
        *byte = byte.wrapping_add(1);
        Ok(MutationResult::Mutated)
    }
}
```

### BytesDecMutator
从 input 中随机选出一个字节，减一
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        let byte = state.rand_mut().choose(input.bytes_mut());
        *byte = byte.wrapping_sub(1);
        Ok(MutationResult::Mutated)
    }
}
```

### ByteNegMutator
从 input 中随机选出一个字节，先按位取反，然后加1
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        let byte = state.rand_mut().choose(input.bytes_mut());
        *byte = (!(*byte)).wrapping_add(1);
        Ok(MutationResult::Mutated)
    }
}
```

### ByteRandMutator
从 input 中随机选出一个字节，然后使用一个 0-254 之间的随机值和它异或
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().is_empty() {
        Ok(MutationResult::Skipped)
    } else {
        let byte = state.rand_mut().choose(input.bytes_mut());
        *byte ^= 1 + state.rand_mut().below(254) as u8;
        Ok(MutationResult::Mutated)
    }
}
```
### Byte/Word/Dword/Qword AddMutator
从 input 中随机选择一个目标大小的串 （Byte/Word/Dword/Qword），然后从下面四个操作里面随机选一个完成
- 加上一个随机值
- 减去一个随机值
- 交换大小端，加上一个随机值，交换大小端
- 交换大小端，减去一个随机值，交换大小端
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().len() < size_of::<$size>() {
        Ok(MutationResult::Skipped)
    } else {
        // choose a random window of bytes (windows overlap) and convert to $size
        let (index, bytes) = state
            .rand_mut()
            .choose(input.bytes().windows(size_of::<$size>()).enumerate());
        let val = <$size>::from_ne_bytes(bytes.try_into().unwrap());

        // mutate
        let num = 1 + state.rand_mut().below(ARITH_MAX) as $size;
        let new_val = match state.rand_mut().below(4) {
            0 => val.wrapping_add(num),
            1 => val.wrapping_sub(num),
            2 => val.swap_bytes().wrapping_add(num).swap_bytes(),
            _ => val.swap_bytes().wrapping_sub(num).swap_bytes(),
        };

        // set bytes to mutated value
        let new_bytes = &mut input.bytes_mut()[index..index + size_of::<$size>()];
        new_bytes.copy_from_slice(&new_val.to_ne_bytes());
        Ok(MutationResult::Mutated)
    }
}
```
实现用到了宏，`$size` 表示串的类型，可能是 `u8/u16/u3/u64`。这里 `ARITH_MAX` 是一个确定的值。
```rust
/// The max value that will be added or subtracted during add mutations
pub const ARITH_MAX: u64 = 35;
```

### Byte/Word/Dword InterestingMutator
从 input 中随机选择一个目标大小的串（Byte/Word/Dword），然后随机使用一个 interesting 的值按大端序或者是小端序替换
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    if input.bytes().len() < size_of::<$size>() {
        Ok(MutationResult::Skipped)
    } else {
        let bytes = input.bytes_mut();
        let upper_bound = (bytes.len() + 1 - size_of::<$size>()) as u64;
        let idx = state.rand_mut().below(upper_bound) as usize;
        let val = *state.rand_mut().choose(&$interesting) as $size;
        let new_bytes = match state.rand_mut().choose(&[0, 1]) {
            0 => val.to_be_bytes(),
            _ => val.to_le_bytes(),
        };
        bytes[idx..idx + size_of::<$size>()].copy_from_slice(&new_bytes);
        Ok(MutationResult::Mutated)
    }
}
```
实现用到了宏，`$size` 表示串的类型，可能是 `u8/u16/u32`。`$interesting` 表示 interesting 的值的数组
```rust
/// Interesting 8-bit values from AFL
pub const INTERESTING_8: [i8; 9] = [-128, -1, 0, 1, 16, 32, 64, 100, 127];
/// Interesting 16-bit values from AFL
pub const INTERESTING_16: [i16; 19] = [
    -128, -1, 0, 1, 16, 32, 64, 100, 127, -32768, -129, 128, 255, 256, 512, 1000, 1024, 4096, 32767,
];
/// Interesting 32-bit values from AFL
pub const INTERESTING_32: [i32; 27] = [
    -128,
    -1,
    0,
    1,
    16,
    32,
    64,
    100,
    127,
    -32768,
    -129,
    128,
    255,
    256,
    512,
    1000,
    1024,
    4096,
    32767,
    -2147483648,
    -100663046,
    -32769,
    32768,
    65535,
    65536,
    100663045,
    2147483647,
];
```

### BytesDeleteMutator
从 input 中删除随机长度的字节串
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size <= 2 {
        return Ok(MutationResult::Skipped);
    }

    let range = rand_range(state, size, size);
    if range.is_empty() {
        return Ok(MutationResult::Skipped);
    }

    input.bytes_mut().drain(range);

    Ok(MutationResult::Mutated)
}
```

### BytesExpandMutator
将 input 扩展 range 长度，然后从 range 的开始位置向前移动剩下的字节串，直到覆盖扩展的长度
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let max_size = state.max_size();
    let size = input.bytes().len();
    if size == 0 || size >= max_size {
        return Ok(MutationResult::Skipped);
    }

    let mut range = rand_range(state, size, min(max_size - size, 16));
    if range.is_empty() {
        return Ok(MutationResult::Skipped);
    }
    let new_size = range.len() + size;

    let mut target = size;
    core::mem::swap(&mut target, &mut range.end);

    input.bytes_mut().resize(new_size, 0);
    input.bytes_mut().copy_within(range, target);

    Ok(MutationResult::Mutated)
}
```
有点复杂举个例子，最后 `[2,3,4,5,6]` 被向前移动 3 格，覆盖 `[0,0,0]`  
```
input [0,1,2,3,4,5,6]
range [2,4]
resized input [0,1,2,3,4,5,6,0,0,0]
result [0,1,2,3,4,2,3,4,5,6]
```

### BytesInsertMutator
向 input 中随机位置插入随机长度重复值的字节串，重复值来自 input 中的随机字节
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let max_size = state.max_size();
    let size = input.bytes().len();
    if size == 0 || size >= max_size {
        return Ok(MutationResult::Skipped);
    }

    let amount = 1 + state.rand_mut().below(min(max_size - size, 16) as u64) as usize;
    let offset = state.rand_mut().below(size as u64 + 1) as usize;

    let val = input.bytes()[state.rand_mut().below(size as u64) as usize];

    input
        .bytes_mut()
        .splice(offset..offset, core::iter::repeat(val).take(amount));

    Ok(MutationResult::Mutated)
}
```

### BytesRandInsertMutator
向 input 中随机位置插入随机长度重复值的字节串，和 `BytesInsertMutator` 不同，重复值是随机的
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let max_size = state.max_size();
    let size = input.bytes().len();
    if size >= max_size {
        return Ok(MutationResult::Skipped);
    }

    let amount = 1 + state.rand_mut().below(min(max_size - size, 16) as u64) as usize;
    let offset = state.rand_mut().below(size as u64 + 1) as usize;

    let val = state.rand_mut().next() as u8;

    input
        .bytes_mut()
        .splice(offset..offset, core::iter::repeat(val).take(amount));

    Ok(MutationResult::Mutated)
}
```

### BytesSetMutator
从 input 中随机选择一个 range，替换为重复值的字节串，重复值来自 input
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size == 0 {
        return Ok(MutationResult::Skipped);
    }
    let range = rand_range(state, size, min(size, 16));
    if range.is_empty() {
        return Ok(MutationResult::Skipped);
    }

    let val = *state.rand_mut().choose(input.bytes());
    let quantity = range.len();
    input
        .bytes_mut()
        .splice(range, core::iter::repeat(val).take(quantity));

    Ok(MutationResult::Mutated)
}
```

### BytesRandSetMutator
从 input 中随机选择一个 range，替换为重复值的字节串，和 `BytesSetMutator` 不同的是，重复值是随机生成的
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size == 0 {
        return Ok(MutationResult::Skipped);
    }
    let range = rand_range(state, size, min(size, 16));
    if range.is_empty() {
        return Ok(MutationResult::Skipped);
    }

    let val = state.rand_mut().next() as u8;
    let quantity = range.len();
    input
        .bytes_mut()
        .splice(range, core::iter::repeat(val).take(quantity));

    Ok(MutationResult::Mutated)
}
```

### BytesCopyMutator
将 input 中的一段字节串拷贝到另一个位置
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size <= 1 {
        return Ok(MutationResult::Skipped);
    }

    let target = state.rand_mut().below(size as u64) as usize;
    let range = rand_range(state, size, size - target);

    input.bytes_mut().copy_within(range, target);

    Ok(MutationResult::Mutated)
}
```

### BytesInsertCopyMutator
从 input 中选出一个字节串，复制并且插入到 input 中
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size <= 1 || size == state.max_size() {
        return Ok(MutationResult::Skipped);
    }

    let target = state.rand_mut().below(size as u64) as usize;
    // make sure that the sampled range is both in bounds and of an acceptable size
    let max_insert_len = min(size - target, state.max_size() - size);
    let range = rand_range(state, size, max_insert_len);

    self.tmp_buf.clear();
    self.tmp_buf.extend(input.bytes()[range].iter().copied());

    input
        .bytes_mut()
        .splice(target..target, self.tmp_buf.drain(..));

    Ok(MutationResult::Mutated)
}
```

### BytesSwapMutator
从 input 中选择两个不相交的字节串，交换
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut I,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    if size <= 1 {
        return Ok(MutationResult::Skipped);
    }

    self.tmp_buf.clear();

    let first = rand_range(state, size, size);
    if state.rand_mut().next() & 1 == 0 && first.start != 0 {
        let second = rand_range(state, first.start, first.start);
        self.tmp_buf.extend(input.bytes_mut().drain(first.clone()));
        self.tmp_buf
            .extend(input.bytes()[second.clone()].iter().copied());
        input
            .bytes_mut()
            .splice(first.start..first.start, self.tmp_buf.drain(first.len()..));
        input.bytes_mut().splice(second, self.tmp_buf.drain(..));
        Ok(MutationResult::Mutated)
    } else if first.end != size {
        let mut second = rand_range(state, size - first.end, size - first.end);
        second.start += first.end;
        second.end += first.end;
        self.tmp_buf.extend(input.bytes_mut().drain(second.clone()));
        self.tmp_buf
            .extend(input.bytes()[first.clone()].iter().copied());
        input.bytes_mut().splice(
            second.start..second.start,
            self.tmp_buf.drain(second.len()..),
        );
        input.bytes_mut().splice(first, self.tmp_buf.drain(..));
        Ok(MutationResult::Mutated)
    } else {
        Ok(MutationResult::Skipped)
    }
```

### CrossoverInsertMutator
从 corpus 中随机选出一个字节串插入到 input 中
```rust
fn mutate(
    &mut self,
    state: &mut S,
    input: &mut S::Input,
    _stage_idx: i32,
) -> Result<MutationResult, Error> {
    let size = input.bytes().len();
    let max_size = state.max_size();
    if size >= max_size {
        return Ok(MutationResult::Skipped);
    }

    // We don't want to use the testcase we're already using for splicing
    let idx = random_corpus_id!(state.corpus(), state.rand_mut());

    if let Some(cur) = state.corpus().current() {
        if idx == *cur {
            return Ok(MutationResult::Skipped);
        }
    }

    let other_size = state
        .corpus()
        .get(idx)?
        .borrow_mut()
        .load_input()?
        .bytes()
        .len();
    if other_size < 2 {
        return Ok(MutationResult::Skipped);
    }

    let range = rand_range(state, other_size, min(other_size, max_size - size));
    let target = state.rand_mut().below(size as u64) as usize;

    let mut other_testcase = state.corpus().get(idx)?.borrow_mut();
    let other = other_testcase.load_input()?;

    input
        .bytes_mut()
        .splice(target..target, other.bytes()[range].iter().copied());

    Ok(MutationResult::Mutated)
}
```

### CrossoverReplaceMutator
从 corpus 中随机选择一个字节串替换 input 中的一个字节串
```rust
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut S::Input,
        _stage_idx: i32,
    ) -> Result<MutationResult, Error> {
        let size = input.bytes().len();
        if size == 0 {
            return Ok(MutationResult::Skipped);
        }

        // We don't want to use the testcase we're already using for splicing
        let idx = random_corpus_id!(state.corpus(), state.rand_mut());
        if let Some(cur) = state.corpus().current() {
            if idx == *cur {
                return Ok(MutationResult::Skipped);
            }
        }

        let other_size = state
            .corpus()
            .get(idx)?
            .borrow_mut()
            .load_input()?
            .bytes()
            .len();
        if other_size < 2 {
            return Ok(MutationResult::Skipped);
        }

        let target = state.rand_mut().below(size as u64) as usize;
        let range = rand_range(state, other_size, min(other_size, size - target));

        let mut other_testcase = state.corpus().get(idx)?.borrow_mut();
        let other = other_testcase.load_input()?;

        input.bytes_mut().splice(
            target..(target + range.len()),
            other.bytes()[range].iter().copied(),
        );

        Ok(MutationResult::Mutated)
    }
```