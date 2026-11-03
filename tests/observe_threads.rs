//! Thread slots of `fastokens::observe` when a process has more threads than
//! slots. Its own test binary (process), so no other test's threads take slots.
#![cfg(feature = "observe")]

use fastokens::observe::{Api, api_call, snapshot};

#[test]
fn more_threads_than_slots_share_the_last_one() {
    let threads = 300;
    for i in 0..threads {
        std::thread::Builder::new()
            .name(format!("many-{i}"))
            .spawn(|| api_call(Api::Decode, 1, 1, 1, false, 0))
            .unwrap()
            .join()
            .unwrap();
    }
    let s = snapshot();
    let slots = s["threads"].as_array().unwrap();
    assert_eq!(slots.len(), 256);
    let last = slots.last().unwrap();
    assert_eq!(last["name"], "(other threads)");
    // Every call is counted exactly once, in a named slot or the shared one.
    let total: u64 = slots.iter().map(|t| t["api_calls"].as_u64().unwrap()).sum();
    assert_eq!(total, threads);
    assert_eq!(last["api_calls"].as_u64().unwrap(), threads - 255);
    assert_eq!(s["api"]["decode"]["calls"].as_u64().unwrap(), threads);
    // Named slots of exited threads are dead; the shared slot never dies.
    assert!(slots[..255].iter().all(|t| t["alive"] == false));
    assert_eq!(slots[0]["name"], "many-0");
}
