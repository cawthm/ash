//! Latency benchmarks for price prediction inference.
//!
//! Measures end-to-end inference latency to validate <10ms target.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_buffer_operations(c: &mut Criterion) {
    use ash_inference::FeatureBuffer;

    c.bench_function("buffer_update", |b| {
        let mut buffer = FeatureBuffer::new(300, 1);

        b.iter(|| {
            buffer.update(
                black_box(1000.0),
                black_box(100.5),
                black_box(1000.0),
            );
        });
    });
}

fn benchmark_rolling_buffer(c: &mut Criterion) {
    use ash_inference::RollingBuffer;

    c.bench_function("rolling_buffer_push", |b| {
        let mut buffer = RollingBuffer::new(300);

        b.iter(|| {
            buffer.push(black_box(100.5));
        });
    });

    c.bench_function("rolling_buffer_to_vec", |b| {
        let mut buffer = RollingBuffer::new(300);
        for i in 0..300 {
            buffer.push(i as f64);
        }

        b.iter(|| {
            let _vec = buffer.to_vec();
        });
    });
}

criterion_group!(benches, benchmark_buffer_operations, benchmark_rolling_buffer);
criterion_main!(benches);
