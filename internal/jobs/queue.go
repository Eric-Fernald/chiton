// Package jobs is an in-process (no Redis) debouncing, coalescing worker queue keyed by string. A key
// never runs concurrently with itself. If re-enqueued mid-run, it runs once more so the final state is seen.
package jobs

import (
	"context"
	"sync"
	"time"
)

// Handler processes one coalesced key. Distinct keys may run concurrently, so it must be safe across
// keys. The queue guarantees a single key never runs concurrently with itself.
type Handler func(ctx context.Context, key string) error

// Options configures a Queue. Zero values fall back to sane defaults.
type Options struct {
	// Workers is the maximum number of keys processed concurrently.
	// Default: 4.
	Workers int
	// Debounce is how long to wait after the last Enqueue of a key before
	// running it. Default: 250ms.
	Debounce time.Duration
	// OnError, if set, is called when a Handler returns a non-nil error.
	OnError func(key string, err error)
	// OnDone, if set, is called after a key's Handler returns (err may be nil).
	// Useful for accumulating "touched" state for a later cluster pass.
	OnDone func(key string, err error)
}

const (
	defaultWorkers  = 4
	defaultDebounce = 250 * time.Millisecond
	drainPoll       = 2 * time.Millisecond
)

// Queue is a debouncing, coalescing, bounded worker queue.
type Queue struct {
	handler  Handler
	debounce time.Duration
	onError  func(string, error)
	onDone   func(string, error)

	// ctx is created in New so Enqueue is safe before Start. Only Stop, Drain, or Start's ctx cancels it.
	ctx    context.Context
	cancel context.CancelFunc

	ready    chan string   // keys whose debounce elapsed, awaiting a worker
	workers  chan struct{} // semaphore bounding concurrent handlers
	draining chan struct{} // closed by Drain to switch dispatch into drain mode
	wg       sync.WaitGroup

	mu      sync.Mutex
	timers  map[string]*time.Timer // keys waiting out their debounce window
	queued  map[string]bool        // keys whose timer fired, not yet picked up by launch
	running map[string]bool        // keys currently executing
	requeue map[string]bool        // keys re-enqueued while running
	stopped bool
}

// New creates a Queue. Call Start to begin processing.
func New(handler Handler, opts Options) *Queue {
	if opts.Workers <= 0 {
		opts.Workers = defaultWorkers
	}
	if opts.Debounce <= 0 {
		opts.Debounce = defaultDebounce
	}
	q := &Queue{
		handler:  handler,
		debounce: opts.Debounce,
		onError:  opts.OnError,
		onDone:   opts.OnDone,
		ready:    make(chan string, 1024),
		workers:  make(chan struct{}, opts.Workers),
		draining: make(chan struct{}),
		timers:   map[string]*time.Timer{},
		queued:   map[string]bool{},
		running:  map[string]bool{},
		requeue:  map[string]bool{},
	}
	q.ctx, q.cancel = context.WithCancel(context.Background())
	return q
}

// Start launches the dispatch loop. Cancelling ctx hard-stops the queue like Stop. Handlers receive the
// queue's own context, which stays live through a graceful Drain.
func (q *Queue) Start(ctx context.Context) {
	q.wg.Add(1)
	go q.dispatch()
	// Bridge an external cancel to a hard stop. For an uncancellable ctx this exits when Stop or Drain
	// cancels q.ctx.
	go func() {
		select {
		case <-ctx.Done():
			q.cancel()
		case <-q.ctx.Done():
		}
	}()
}

// Enqueue schedules key to run after the debounce window. Repeated calls within
// the window coalesce. Safe for concurrent use, including before Start.
func (q *Queue) Enqueue(key string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.stopped {
		return
	}
	if q.running[key] {
		// Already running → run exactly once more after it finishes.
		q.requeue[key] = true
		return
	}
	if q.queued[key] {
		// Its timer fired and the run has not started, so that run sees this change.
		return
	}
	if t, ok := q.timers[key]; ok {
		if t.Stop() {
			// Cancelled before it fired: release its Add and re-arm fresh below,
			// extending the debounce window.
			q.wg.Done()
			delete(q.timers, key)
		} else {
			// The timer fired but its callback has not taken q.mu yet, so the key is not queued and the
			// run it leads to sees this change. Re-arming would double-dispatch (and double-Done the
			// WaitGroup).
			return
		}
	}
	q.armLocked(key)
}

// armLocked starts a debounce timer for key. The caller holds q.mu. Each arm's wg.Add is balanced by
// exactly one wg.Done, from either the callback's defer or a timer Stop() that returns true.
func (q *Queue) armLocked(key string) {
	q.wg.Add(1)
	q.timers[key] = time.AfterFunc(q.debounce, func() {
		defer q.wg.Done()
		// Swap the timer entry for a queued marker under one lock, before the send. A timer entry must
		// not outlive the dispatch: Enqueue reads a fired timer's entry as "a run is still coming", and
		// if that run had already finished the change was dropped. The marker also keeps Drain's
		// idle-check from seeing the key as neither waiting nor queued.
		q.mu.Lock()
		delete(q.timers, key)
		q.queued[key] = true
		q.mu.Unlock()
		select {
		case q.ready <- key:
		case <-q.ctx.Done():
		}
	})
}

// dispatch pulls ready keys and runs them under the worker semaphore.
func (q *Queue) dispatch() {
	defer q.wg.Done()
	for {
		select {
		case <-q.ctx.Done():
			return // hard stop: abandon queued work
		case <-q.draining:
			q.drainLoop()
			return
		case key := <-q.ready:
			q.launch(key)
		}
	}
}

// drainLoop processes every outstanding key (queued and not-yet-fired timers
// that Drain hands over) until the pipeline is fully quiescent, then returns.
func (q *Queue) drainLoop() {
	for {
		select {
		case key := <-q.ready:
			q.launch(key)
			continue
		default:
		}
		q.mu.Lock()
		// Re-feed keys re-enqueued while their handler ran, skipping any still in flight (their run()
		// keeps the marker for a later pass).
		var requeued []string
		for k := range q.requeue {
			if q.running[k] {
				continue
			}
			requeued = append(requeued, k)
			delete(q.requeue, k)
		}
		idle := len(q.timers) == 0 && len(q.queued) == 0 && len(q.running) == 0 && len(q.requeue) == 0
		q.mu.Unlock()

		for _, k := range requeued {
			select {
			case q.ready <- k:
			case <-q.ctx.Done():
			}
		}
		if idle && len(requeued) == 0 && len(q.ready) == 0 {
			return
		}
		select {
		case <-q.ctx.Done():
			return
		case <-time.After(drainPoll):
		}
	}
}

// launch acquires a worker slot and runs key in a fresh goroutine, unless the key is already in flight,
// in which case it requeues the key rather than running it twice concurrently.
func (q *Queue) launch(key string) {
	select {
	case q.workers <- struct{}{}:
	case <-q.ctx.Done():
		return
	}
	q.mu.Lock()
	delete(q.queued, key)
	if q.running[key] {
		// The queued marker keeps a key from reaching ready twice. If one does anyway, requeue it (run()
		// consumes the marker) and release the slot, keeping the key serialized.
		q.requeue[key] = true
		q.mu.Unlock()
		<-q.workers
		return
	}
	q.running[key] = true
	q.mu.Unlock()
	q.wg.Add(1)
	go func(k string) {
		defer q.wg.Done()
		defer func() { <-q.workers }()
		q.run(k)
	}(key)
}

// run executes the handler for one key and re-enqueues it if it was touched
// while running.
func (q *Queue) run(key string) {
	err := q.handler(q.ctx, key)
	if err != nil && q.onError != nil {
		q.onError(key, err)
	}
	if q.onDone != nil {
		q.onDone(key, err)
	}

	q.mu.Lock()
	delete(q.running, key)
	again := q.requeue[key]
	stopped := q.stopped
	if !stopped {
		// Normal path: consume the marker and re-arm via Enqueue below.
		delete(q.requeue, key)
	}
	// Once stopped, keep the marker so drainLoop re-feeds the key and its final state is processed.
	// Stop also sets stopped but cancels the context, so its leftover markers are never flushed.
	q.mu.Unlock()

	if again && !stopped {
		q.Enqueue(key)
	}
}

// Drain stops accepting new work, processes everything already enqueued (including unfired timers),
// and waits for all handlers before shutting down. A second call is a no-op.
func (q *Queue) Drain() {
	q.mu.Lock()
	if q.stopped {
		q.mu.Unlock()
		return
	}
	q.stopped = true
	var pending []string
	for key, t := range q.timers {
		if t.Stop() {
			q.wg.Done() // balance the arming Add
			delete(q.timers, key)
			pending = append(pending, key) // still owed a run
		}
		// Stop()==false: the callback already fired; it will queue the key to
		// ready and delete its own timer entry, so drainLoop will process it.
	}
	q.mu.Unlock()

	// Hand the cancelled-but-pending keys to ready, then switch dispatch into
	// drain mode (push first so drainLoop's idle-check never fires early).
	for _, key := range pending {
		select {
		case q.ready <- key:
		case <-q.ctx.Done():
		}
	}
	close(q.draining)

	q.wg.Wait()
	q.cancel()
}

// Stop stops accepting new work, abandons pending/queued work, cancels in-flight
// handlers' context, and waits for them to return. Safe to call once.
func (q *Queue) Stop() {
	q.mu.Lock()
	if q.stopped {
		q.mu.Unlock()
		return
	}
	q.stopped = true
	for key, t := range q.timers {
		if t.Stop() {
			q.wg.Done()
		}
		delete(q.timers, key)
	}
	q.mu.Unlock()

	q.cancel()
	q.wg.Wait()
}
