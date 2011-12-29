#!/bin/bash

source $( dirname $BASH_SOURCE )/__lib.sh || exit 1

validate_farm_env

############################################
#
# DEBUG SETTINGS

ZOMBIE_SCREENS=false
#
############################################


DISPATCHERSTATE_DIR=$( pwenv dispatcherstate_dir ) || exit 1
require "$DISPATCHERSTATE_DIR" "dispatcher dir cannot be empty!!!"
mkdir -p "$DISPATCHERSTATE_DIR" || exit 1

export DISPATCHER_SCREEN_SESSION="____pwfarm_dispatcher__farm_$( pwenv farmname )__session_$( pwenv sessionname )____"
MUTEX=$DISPATCHERSTATE_DIR/mutex
PARMS=$DISPATCHERSTATE_DIR/parms
TASKS=$DISPATCHERSTATE_DIR/tasks
PID=$DISPATCHERSTATE_DIR/pid
FIELDNUMBERS=$DISPATCHERSTATE_DIR/fieldnumbers

BLOB_DIR=${DISPATCHERSTATE_DIR}/blob
BLOB_LOCAL=${BLOB_DIR}/blob.zip
BLOB_REMOTE="~/__pwfarm_blob__user_$( pwenv pwuser )__farm_$( pwenv farmname )__session_$( pwenv sessionname ).zip"
BROADCAST_COMPLETE=$DISPATCHERSTATE_DIR/broadcast_complete
FIELD_BORN=$DISPATCHERSTATE_DIR/field_born

PWUSER=$( pwenv pwuser )
OSUSER=$( pwenv osuser )

function screen_active()
{
    screen -ls | grep "\\b${DISPATCHER_SCREEN_SESSION}\\b" > /dev/null
}

function init_screen()
{
    screen -d -m -S "${DISPATCHER_SCREEN_SESSION}"

    if $ZOMBIE_SCREENS; then
	while ! screen_active; do
	    wait 0.1
	done

	screen -S "${DISPATCHER_SCREEN_SESSION}" -X zombie kr
    fi
}

function resume_screen()
{
    screen -r -S "${DISPATCHER_SCREEN_SESSION}"
}

function kill_screen()
{
    while screen_active; do
	screen -S "${DISPATCHER_SCREEN_SESSION}" -X quit
	sleep 1
    done
}

mode="$1"

prompt_password=false
broadcast=false

###
### Mode-specific logic prior to interfacing with field nodes
###
case "$mode" in
    "dispatch")
	payload=$( canonpath "$2" )
	shift 2
	tasks="$@"

	[ ! -z "$tasks" ] || err "No tasks given to dispatcher!"

	if [ ! -z "$( for x in $tasks; do taskmeta get $x id; done | sort | uniq -d )" ]; then
	    err "Duplicate task IDs found"
	fi

	if for x in $tasks; do taskmeta get $x sudo; done | grep "true" > /dev/null; then
	    prompt_password=true
	fi

	for x in $tasks; do
	    assert taskmeta validate $x
	    if taskmeta has $x required_field; then
		required_field="$(taskmeta get $x required_field)"
		if ! pwenv fieldnumbers | grep "\b${required_field}\b" > /dev/null; then
		    err "Task requires field $required_field! fields=$(pwenv fieldnumbers)"
		fi
	    fi
	done

	mutex_lock $MUTEX

	if [ -e "$PARMS" ]; then
	    mutex_unlock $MUTEX
	    err "You're either already running a task in this session or need to 'clear/recover'. (Exists: $PARMS)"
	fi
	if screen_active ; then
	    mutex_unlock $MUTEX
	    err "You must be running a task in this session. (Dispatcher screen already active!)"
	fi

	rm -rf $TASKS
	mkdir -p $TASKS/pending
	mkdir -p $TASKS/running
	mkdir -p $TASKS/complete

	BATCHID=$(hostname)_${RANDOM}_$(date | sed -e "s/ /_/g" -e "s/:/./g")

	for x in $tasks; do
	    path=$TASKS/pending/$( taskmeta get $x id )
	    cp $x $path
	    taskmeta set $path batchid $BATCHID
	done

	if ! init_screen; then
	    mutex_unlock $MUTEX
	    err "Failed initing dispatcher screen!"
	fi

	rm -f $BROADCAST_COMPLETE

	(
	    echo $prompt_password
	    echo $payload
	) > $PARMS

	echo $$ > $PID
	echo $( pwenv fieldnumbers ) > $FIELDNUMBERS

	broadcast="true"

	mutex_unlock $MUTEX
	;;
    "recover")
	mutex_lock $MUTEX

	if [ ! -e "$PARMS" ]; then
	    mutex_unlock $MUTEX
	    err "Can't find job to recover. Is this the right session? (Does not exist: $PARMS)"
	fi
	if [ ! -e "$FIELDNUMBERS" ]; then
	    mutex_unlock $MUTEX
	    err "Can't find field numbers to recover. (Does not exist: $FIELDNUMBERS)"
	fi
	if screen_active ; then
	    mutex_unlock $MUTEX
	    err "Dispatcher screen already active! You must be running a job in this session."
	fi
	if [ -e "$PID" ]; then
	    if ps -e | grep "\\b$( cat $PID )\\b"; then
		mutex_unlock $MUTEX
		err "Dispatcher already alive!"
	    fi
	fi
	if ! init_screen; then
	    mutex_unlock $MUTEX
	    err "Failed initing dispatcher screen!"
	fi

	function parm()
	{
	    head -n $(( $1 + 1 )) "$PARMS" | tail -n 1
	}

	prompt_password=$( parm 0 )
	payload=$( parm 1 )

	__pwfarm_config env set fieldnumbers $( fieldnums_from_hostnames $(cat $FIELDNUMBERS) )

	echo $$ > $PID

	broadcast="true"

	mutex_unlock $MUTEX
	;;
    "task_get")
	mutex_lock $MUTEX

	path_result="$2"
	rm -f $path_result

	path_task=""

	fieldnumber=$( pwenv fieldnumber )

	function __sorted_tasks()
	{
	    local dir=$TASKS/$1
	    ls $dir/* 2>/dev/null |
	    while read x; do basename $x; done |
	    sort -n |
	    while read x; do echo $dir/$x; done
	}

	pending=$( __sorted_tasks pending )
	running=$( __sorted_tasks running )

	# we might be recovering from a crash
	for x in $running $pending; do
	    if [ "$( taskmeta get $x assigned )" == $fieldnumber ]; then
		path_task=$x
		break;
	    fi
	done

	if [ -z "$path_task" ]; then
	    # try to find task with field required_field
	    for x in $pending; do
		if [ "$( taskmeta get $x required_field )" == $fieldnumber ]; then
		    path_task=$x
		    break;
		fi
	    done
	fi

	if [ -z "$path_task" ]; then
	    # any task will do, as long as it's not assigned and doesn't require a specific field
	    for x in $pending; do
		if ! taskmeta has $x assigned && ! taskmeta has $x required_field; then
		    path_task=$x
		    break;
		fi
	    done
	fi

	if [ ! -z "$path_task" ]; then
	    taskmeta set $path_task assigned $fieldnumber
	    cp $path_task $path_result
	    if [ $(dirname $path_task) != $TASKS/running ]; then
		mv $path_task $TASKS/running
	    fi
	fi

	mutex_unlock $MUTEX

	if [ -e $path_result ]; then
	    exit 0
	else
	    exit 1
	fi
	;;
    "task_done")
	mutex_lock $MUTEX

	path_task=$2

	fieldnumber=$( pwenv fieldnumber )
	taskmeta set $path_task assigned $fieldnumber

	taskid=$( taskmeta get $path_task id )

	exitval=0

	if [ -e $TASKS/running/$taskid ]; then
	    cp $path_task $TASKS/complete/$taskid
	    rm $TASKS/running/$taskid
	elif [ ! -e $TASKS/complete/$taskid ]; then
	    # it might be in complete already if we're recovering from a system crash.
	    warn "task_done invoked for $taskid, but it's not in running or complete!"
	    exitval=1
	fi

	mutex_unlock $MUTEX

	exit $exitval
	;;
    "clear")
	if [ "$2" == "--exit" ]; then
	    # We're being invoked as part of the exit procedure
	    echo "Performing healthy pwfarm state clear."
	    shift

	    mutex_lock $MUTEX;
	else
	    # This is an abort/force-clear operation.
	    echo "Forcing pwfarm state clear. This could take a while."

	    # If we can't get the mutex, we still need to keep going.
	    mutex_trylock $MUTEX;

	    if [ -e $PID ]; then
		pid=$( cat $PID )
		kill $pid  2>/dev/null
	    fi

	    kill_screen

	    # Force clear of all machines on farm.
	    pwquery fieldnumbers $(pwenv farmname) > $FIELDNUMBERS
	fi
	rm -f "$PARMS"
	rm -f "$PID"
	rm -f "$BROADCAST_COMPLETE"
	rm -rf "$BLOB_DIR"
	;;
    "disconnect")
	mutex_lock $MUTEX;

	if [ "$2" != "--exit" ]; then
	    if is_process_alive $PID; then
		echo "Killing dispatcher process..." >&2
		pid=$( cat $PID )
		kill $pid  2>/dev/null
	    else
		echo "No dispatcher process detected" >&2
	    fi
	fi

	if screen_active ; then
	    echo "Killing dispatcher screen..." >&2
	    kill_screen
	else
	    echo "No dispatcher screen detected" >&2
	fi
	;;
    "exit")
	succeeded=""
	failed=""
	exitval=0

	for x in $( find $TASKS/complete/ -type f | sort -n ); do
	    field=$( fieldhostname_from_num $(taskmeta get $x assigned) )
	    statusid="$( taskmeta get $x statusid )"
	    if [ ! -z "$statusid" ]; then
		name="$field/$statusid"
	    else
		name="$field"
	    fi

	    if [ "$(taskmeta get $x exitval)" != "0" ]; then
		failed="${failed}${name} "
		exitval=1
	    else
		succeeded="${succeeded}${name} "
	    fi
	done
	;;
    *)
	err "Invalid mode: $mode"
	;;
esac

###
### Prompt user for sudo password if needed
###
if $prompt_password; then
    PASSWORD=""
    
    while [ -z "$PASSWORD" ]; do
        # turn off echo for reading password
	stty -echo
	read -p "Please enter password of administrator on farm machines (for sudo): " PASSWORD
	echo
        # turn echo back on
	stty echo
    done
else
    PASSWORD="nil"
fi

###
### Broadcast blob to field nodes if needed
###
if $broadcast; then
    if [ ! -e "$BROADCAST_COMPLETE" ]; then
	require "$BLOB_DIR" "blob dir can't be null!"
	rm -rf "$BLOB_DIR"

	set -e
	mkdir -p "$BLOB_DIR"
	pushd_quiet .
	cd "$BLOB_DIR"
	mkdir tasks
	cp $TASKS/pending/* tasks
        mkdir payload
	cp "$payload" payload/payload.zip
	mkdir scripts
	cp "$PWFARM_SCRIPTS_DIR/__lib.sh" scripts
	cp "$PWFARM_SCRIPTS_DIR/__pwfarm_field.sh" scripts
	cp "$PWFARM_SCRIPTS_DIR/__pwfarm_taskmeta.sh" scripts
	cp "$PWFARM_SCRIPTS_DIR/__pwfarm_runutil.sh" scripts
	cp "$PWFARM_SCRIPTS_DIR/__pwfarm_config.sh" scripts
	cp "$PWFARM_SCRIPTS_DIR/__pwfarm_status.py" scripts
	zip -qr "$BLOB_LOCAL" *
	popd_quiet
	set +e

	#
	# Broadcast blob to fields
	#
	$PWFARM_SCRIPTS_DIR/__pwfarm_broadcast.sh "$BLOB_LOCAL" "$BLOB_REMOTE"

	touch "$BROADCAST_COMPLETE"
    fi
fi

###
### Perform task on all field nodes
###
screen_window=1

if [ -e $FIELDNUMBERS ]; then
    for fieldnumber in $( cat $FIELDNUMBERS ); do

	fieldhostname=$( fieldhostname_from_num $fieldnumber )

	FARMER_SH="$PWFARM_SCRIPTS_DIR/__pwfarm_farmer.sh"

	if [ "$mode" == "dispatch" ] || [ "$mode" == "recover" ]; then
	    rm -f $FIELD_BORN

	    title="$fieldhostname - $command"
	    screen -S "${DISPATCHER_SCREEN_SESSION}" -X screen -t "$title" \
		"$FARMER_SH" \
		$fieldnumber \
                $mode \
		$screen_window \
		$FIELD_BORN \
                "${BLOB_REMOTE}" \
                "$PASSWORD"

	    while [ ! -e $FIELD_BORN ]; do
		sleep 0.1
	    done

	    screen_window=$(( $screen_window + 1 ))
	else
	    case "$mode" in 
		"clear")
		    echo "   [ Clearing $fieldnumber (farm=$(pwenv farmname), session=$(pwenv sessionname)) ]"
		    ( $FARMER_SH $fieldnumber clear $BLOB_REMOTE)&
		    ;;

		"disconnect")
		    echo "   [ Disconnecting $fieldnumber... ]"
		    $FARMER_SH $fieldnumber disconnect
		    ;;

		"exit")
		    # no-op
		    ;;
		*)
		    err "Invalid mode: $mode"
		    ;;
	    esac
	fi
    done
fi

###
### Mode-specific logic after interfacing with field nodes
###
if [ "$mode" == "dispatch" ] || [ "$mode" == "recover" ]; then
    win0pid=$DISPATCHERSTATE_DIR/win0pid
    rm -f win0pid
    # kill window 0. there's a race in just using -X kill and then attaching, so we "stuff" a command into the window.
    # it writes its pid to a file, then exits... and we can wait for it to actually die
    screen -S "${DISPATCHER_SCREEN_SESSION}" -p 0 -X stuff 'echo $$>'$win0pid'; exit'$'\n'

    # wait for win0 to write its pid
    echo "Waiting for win0 to write pid"
    while [ ! -e $win0pid ]; do sleep 0.1; done

    # wait for win0 to die
    echo "Waiting for win0 to die"
    while is_process_alive $win0pid; do sleep 0.1; done
    rm $win0pid

    # sleep another 1 second, just for good measure
    sleep 1

    # bring up screen, starting at the windowlist
    echo "Attaching to dispatcher screen..."
    screen -S "${DISPATCHER_SCREEN_SESSION}" -p = -r
    while screen_active; do
	echo
	echo "You have detached the dispatcher screen."
	echo
        echo "  (r)esume"
	echo "  (c)lear"
	echo "  (d)isconnect"
	echo
	if read -p "Enter command: " -t 10 choice; then
	    case "$choice" in
		c)
		    $0 clear --exit
		    reset
		    exit 1
		    ;;
		d)
		    $0 disconnect --exit
		    reset
		    exit 1
		    ;;
	    esac
	fi

	resume_screen
    done

    $0 "exit"
else
    case "$mode" in
	"clear")
	    wait
	    rm -rf "$DISPATCHERSTATE_DIR" # this also unlocks MUTEX
	    ;;

	"disconnect")
	    mutex_unlock $MUTEX
	    echo "dispatcher disconnect complete." >&2
	    ;;

	"exit")
	    $0 "clear" --exit
	    echo -n "pwfarm task complete. SUCCESS="
	    if [ "$exitval" == "0" ]; then
		echo "TRUE"
	    else
		echo "FALSE"

		if [ ! -z "$succeeded" ]; then
		    echo "succeeded=$succeeded"
		fi
		if [ ! -z "$failed" ]; then
		    echo "failed=$failed"
		fi
	    fi

	    exit $exitval
	    ;;
    esac
fi