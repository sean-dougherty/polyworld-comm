#include "Mutex.h"

#include <assert.h>
#include "misc.h"

WaitMutex::WaitMutex()
{
	int rc = pthread_mutex_init( &mutex, NULL );
	require( rc == 0 );
}

WaitMutex::~WaitMutex()
{
	pthread_mutex_destroy( &mutex );
}

void WaitMutex::lock()
{
	int rc = pthread_mutex_lock( &mutex );
	require( rc == 0 );
}

void WaitMutex::unlock()
{
	int rc = pthread_mutex_unlock( &mutex );
	require( rc == 0 );
}

#if defined(SPINLOCK_PTHREAD)
SpinMutex::SpinMutex()
{
	int rc = pthread_spin_init( &spinlock, PTHREAD_PROCESS_PRIVATE );
	require( rc == 0 );
}

SpinMutex::~SpinMutex()
{
	pthread_spin_destroy( &spinlock );
}

void SpinMutex::lock()
{
	int rc = pthread_spin_lock( &spinlock );
	require( rc == 0 );
}

void SpinMutex::unlock()
{
	int rc = pthread_spin_unlock( &spinlock );
	require( rc == 0 );
}
#elif defined(SPINLOCK_APPLE)
SpinMutex::SpinMutex()
{
	OSSpinLockUnlock( &spinlock );
}

SpinMutex::~SpinMutex()
{
}

void SpinMutex::lock()
{
	OSSpinLockLock( &spinlock );
}

void SpinMutex::unlock()
{
	OSSpinLockUnlock( &spinlock );
}
#endif // SPINLOCK_*

ConditionMonitor::ConditionMonitor()
: WaitMutex()
{
	int rc = pthread_cond_init( &cond, NULL );
	require( rc == 0 );
}

ConditionMonitor::~ConditionMonitor()
{
	pthread_cond_destroy( &cond );
}

void ConditionMonitor::notify()
{
	int rc = pthread_cond_signal( &cond );
	require( rc == 0 );
}

void ConditionMonitor::notifyAll()
{
	int rc = pthread_cond_broadcast( &cond );
	require( rc == 0 );
}

void ConditionMonitor::wait()
{
	int rc = pthread_cond_wait( &cond, &mutex );
	require( rc == 0 );
}

MutexGuard::MutexGuard( IMutex *mutex )
{
	init( mutex );
}

MutexGuard::MutexGuard( IMutex &mutex )
{
	init( &mutex );
}

MutexGuard::~MutexGuard()
{
	mutex->unlock();
}

void MutexGuard::init( IMutex *mutex )
{
	this->mutex = mutex;

	mutex->lock();
}
