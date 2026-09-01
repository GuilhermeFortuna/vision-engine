use std::fs;
use std::thread;
use std::time::Duration;

use vision_engine::pipeline::{
    FaultConfig, Pipeline, QUEUE_CAPACITY,
    test_support::{sample_model, test_config},
};

const STALL_DURATION: Duration = Duration::from_secs(2);

fn resident_memory_kb() -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next()) {
            (Some("VmRSS:"), Some(value)) => value.parse().ok(),
            _ => None,
        }
    })
}

fn thread_count() -> usize {
    fs::read_dir("/proc/self/task")
        .map(|entries| entries.count())
        .unwrap_or(0)
}

fn assert_queues_within_capacity(pipeline: &Pipeline) {
    let depths = pipeline.queue_depths();
    let peaks = pipeline.peak_queue_depths();
    for (name, (depth, capacity), (peak, peak_capacity)) in [
        ("decoded", depths.decoded, peaks.decoded),
        ("prepared", depths.prepared, peaks.prepared),
        ("detected", depths.detected, peaks.detected),
        ("tracked", depths.tracked, peaks.tracked),
    ] {
        assert_eq!(capacity, QUEUE_CAPACITY, "{name} capacity mismatch");
        assert_eq!(
            peak_capacity, QUEUE_CAPACITY,
            "{name} peak capacity mismatch"
        );
        assert!(
            depth <= capacity,
            "{name} depth {depth} exceeded capacity {capacity}"
        );
        assert!(
            peak <= capacity,
            "{name} peak depth {peak} exceeded capacity {capacity}"
        );
    }
}

#[test]
fn stalled_decode_applies_backpressure_and_shuts_down_cleanly() {
    stalled_stage_applies_backpressure_and_shuts_down_cleanly("decode");
}

#[test]
fn stalled_preprocess_applies_backpressure_and_shuts_down_cleanly() {
    stalled_stage_applies_backpressure_and_shuts_down_cleanly("preprocess");
}

#[test]
fn stalled_infer_applies_backpressure_and_shuts_down_cleanly() {
    stalled_stage_applies_backpressure_and_shuts_down_cleanly("infer");
}

#[test]
fn stalled_track_applies_backpressure_and_shuts_down_cleanly() {
    stalled_stage_applies_backpressure_and_shuts_down_cleanly("track");
}

fn stalled_stage_applies_backpressure_and_shuts_down_cleanly(stage: &str) {
    let Some(config) = test_config(250) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");

    let rss_before = resident_memory_kb().expect("VmRSS should be readable");
    let mut frames_consumed = 0_u64;

    while frames_consumed < 5 {
        if pipeline.next_tracked().is_some() {
            frames_consumed += 1;
        } else {
            break;
        }
    }

    pipeline.stall_stage_for(stage, STALL_DURATION);
    let stall_started = std::time::Instant::now();

    while stall_started.elapsed() < STALL_DURATION {
        assert_queues_within_capacity(&pipeline);
        let rss_now = resident_memory_kb().expect("VmRSS should be readable");
        let allowance = rss_before / 10 + 50_240;
        assert!(
            rss_now <= rss_before + allowance,
            "resident memory grew from {rss_before} kB to {rss_now} kB during {stage} stall"
        );
        thread::sleep(Duration::from_millis(50));
        let _ = pipeline.next_tracked();
    }

    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("shutdown during stall should join every stage thread");
}

#[test]
fn shutdown_with_full_queues_joins_all_stage_threads() {
    let Some(config) = test_config(200) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");
    thread::sleep(Duration::from_millis(100));
    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("shutdown with unconsumed queues should join cleanly");
}

#[test]
fn shutdown_with_empty_queues_joins_all_stage_threads() {
    let Some(config) = test_config(50) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");
    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("shutdown before first frame should join cleanly");
}

#[test]
fn shutdown_mid_frame_joins_all_stage_threads() {
    let Some(config) = test_config(100) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");
    for _ in 0..10 {
        assert!(pipeline.next_tracked().is_some());
    }
    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("mid-frame shutdown should join cleanly");
}

#[test]
fn shutdown_at_end_of_input_joins_all_stage_threads() {
    let Some(config) = test_config(30) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");
    while pipeline.next_tracked().is_some() {}
    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("end-of-input shutdown should join cleanly");
}

#[test]
fn shutdown_before_first_frame_joins_all_stage_threads() {
    let Some(config) = test_config(50) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let pipeline =
        Pipeline::spawn_for_test(&config, model, FaultConfig::default()).expect("pipeline spawn");
    pipeline.request_shutdown();
    pipeline
        .join()
        .expect("startup shutdown should join cleanly");
}

#[test]
fn repeated_start_stop_cycles_leave_stable_threads_and_memory() {
    let Some(config) = test_config(40) else {
        eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
        return;
    };

    let mut post_join_threads = None;
    let mut post_join_rss = None;

    for cycle in 0..10 {
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn_for_test(&config, model, FaultConfig::default())
            .expect("pipeline spawn");
        for _ in 0..20 {
            let _ = pipeline.next_tracked();
        }
        pipeline.request_shutdown();
        pipeline
            .join()
            .unwrap_or_else(|err| panic!("cycle {cycle} failed to join: {err:#}"));

        let threads_after_join = thread_count();
        let rss_after_join = resident_memory_kb().expect("VmRSS should be readable");
        if let Some(baseline) = post_join_threads {
            assert!(
                threads_after_join <= baseline + 2,
                "cycle {cycle} thread count {threads_after_join} exceeded post-join baseline {baseline}"
            );
        } else {
            post_join_threads = Some(threads_after_join);
        }

        if let Some(baseline_rss) = post_join_rss {
            let allowance = baseline_rss / 10 + 50_240;
            assert!(
                rss_after_join <= baseline_rss + allowance,
                "cycle {cycle} RSS {rss_after_join} kB exceeded post-join baseline {baseline_rss} kB + allowance {allowance} kB"
            );
        } else {
            post_join_rss = Some(rss_after_join);
        }
    }
}
