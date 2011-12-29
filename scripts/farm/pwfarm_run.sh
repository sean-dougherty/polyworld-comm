#!/bin/bash

if [ -z "$PWFARM_SCRIPTS_DIR" ]; then
    source $( dirname $BASH_SOURCE )/__pwfarm_runutil.sh || exit 1
else
    source $PWFARM_SCRIPTS_DIR/__pwfarm_runutil.sh || exit 1
fi

function usage()
{
    cat >&2 <<EOF
usage: $( basename $0 ) [-w:p:N:a:f:o:c:z:] run_id

ARGS:

   run_id         Unique ID for run.

OPTIONS:

   -w worldfile
                  Path of local worldfile to be executed. If not provided, it is
                assumed a run already exists on the farm of run_id.

   -p parms_overlay
                  Path of local worldfile overlay file. Allows for setting parameters
                on a per-run basis.

   -N num_runs
                  Specify number of runs, allowing more runs than machines on farm.
                Only legal with -w. Not legal with -p.

   -a analysis_script
                  Path of script that is to be executed after simulation.

   -F fetch_list
                  Files to be pulled back from run. For example:

                    -F "stat/* *.wf"

                  By default, the following is fetched:

                    $DEFAULT_FETCH_LIST

   -f fields
                  Specify fields on which this should run. Must be a single argument,
                so use quotes. e.g. -f "0 1" or -f "{0..3}"

   -o run_owner
                  Specify owner of run.

   -c config_script
                  Path of script that is to be executed prior to running 
                simulation.

   -z input_zip
                  Path of zip file that is to be sent to each machine.

EOF

    if [ ! -z "$1" ]; then
	echo >&2
	err $*
    fi

    exit 1
}

if [ -z "$1" ]; then
    usage
fi

if [ "$1" == "--field" ]; then
    field=true
else
    field=false
fi

if ! $field; then
    validate_farm_env

    ########################
    ###                  ###
    ### EXECUTE ON LOCAL ###
    ###                  ###
    ########################

    WORLDFILE="nil"
    OVERLAY="nil"
    NRUNS="nil"
    PRERUN="nil"
    POSTRUN="nil"
    INPUT_ZIP="nil"
    OWNER=$( pwenv pwuser )
    FETCH_LIST="$DEFAULT_FETCH_LIST"

    while getopts "w:p:N:a:f:o:c:z:F:" opt; do
	case $opt in
	    w)
		WORLDFILE="$OPTARG"
		;;
	    p)
		OVERLAY="$OPTARG"
		;;
	    N)
		NRUNS="$OPTARG"
		is_integer "$NRUNS" || err "-N value must be integer"
		[ "$NRUNS" -gt "0" ] || err "-N value must be > 0"
		;;
	    a)
		POSTRUN="$OPTARG"
		;;
	    f)
		__pwfarm_config env set fieldnumbers "$OPTARG"
		validate_farm_env
		;;
	    o)
		OWNER="$OPTARG"
		;;
	    c)
		PRERUN="$OPTARG"
		;;
	    z)
		INPUT_ZIP="$OPTARG"
		;;
	    F)
		FETCH_LIST="$OPTARG"
		;;
	    *)
		exit 1
		;;
	esac
    done

    if [ "$NRUNS" != "nil" ]; then
	[ "$WORLDFILE" != "nil" ] || err "-N requires -w"
	[ "$OVERLAY" == "nil" ] || err "-N incompatible with -p"
    fi

    if [ "$OVERLAY" != "nil" ]; then
	[ "$WORLDFILE" != "nil" ] || err "-p requires -w"
    fi

    shift $(( $OPTIND - 1 ))
    if [ $# -lt 1 ]; then
	usage "Missing arguments"
    elif [ $# -gt 1 ]; then
	shift
	usage "Unexpected arguments: $*"
    fi

    RUNID="$( normalize_runid "$1" )"
    validate_runid "$RUNID"

    TMP_DIR=$( create_tmpdir )
    PAYLOAD_DIR=$TMP_DIR/payload
    TASKS_DIR=$TMP_DIR/tasks
    RUN_PACKAGE_DIR=$PAYLOAD_DIR/run_package

    mkdir -p $PAYLOAD_DIR
    mkdir -p $TASKS_DIR
    mkdir -p $RUN_PACKAGE_DIR

    TASKS=""

    if [ "$WORLDFILE" != "nil" ]; then
	if [ "$OVERLAY" != "nil" ]; then
	    ntasks=$( proputil len "$OVERLAY" overlays )

	    mkdir -p $TMP_DIR/overlay

	    # Verify we can apply the overlay, for catching errors quickly.
	    for (( i=0; i < $ntasks; i++ )); do
		(
		    success=false
		    if proputil overlay "$WORLDFILE" "$OVERLAY" $i >$TMP_DIR/overlay/$i; then
			if proputil apply $POLYWORLD_DIR/default.wfs $TMP_DIR/overlay/$i >/dev/null; then
			    success=true
			fi
		    fi
		    if ! $success; then
			touch $TMP_DIR/overlay/fail
		    fi
		) &
	    done

	    [ ! -e $TMP_DIR/overlay/fail ] || err "Invalid overlay"
	elif [ "$NRUNS" != "nil" ]; then
	    ntasks=$NRUNS
	else
	    ntasks=$( len $(pwenv fieldnumbers) )
	fi

	#
	# Define Worldfile Tasks
	#
	for (( taskid=0; taskid < $ntasks; taskid++ )); do
	    path=$TASKS_DIR/$taskid
	    TASKS="$TASKS $path"

	    taskmeta set $path id $taskid
	    taskmeta set $path nid $taskid
	done
    else
	#
	# No Worldfile
	#
	if [ ! -e "$(stored_run_path_local $OWNER $RUNID "0")/.pwfarm" ]; then
	    err "Cannot find local run data. Please fetch run data."
	fi

	#
	#  Define Non-Worldfile Tasks
	#
	taskid=0

	fieldnumbers=$( pwenv fieldnumbers )

	for run in $(stored_run_path_local $OWNER $RUNID "*"); do
	    assert [ -e $run/.pwfarm/fieldnumber ]
	    assert [ -e $run/.pwfarm/nid ]

	    fieldnumber=$(cat $run/.pwfarm/fieldnumber)

	    if contains $fieldnumbers $fieldnumber; then
		nid=$(cat $run/.pwfarm/nid)
		assert [ "run_$nid" == "$(basename $run)" ]

		path=$TASKS_DIR/$taskid
		TASKS="$TASKS $path"

		taskmeta set $path id $taskid
		taskmeta set $path required_field $fieldnumber
		taskmeta set $path nid $nid

		taskid=$(( $taskid + 1 ))
	    fi

	done
    fi

    #
    # Set Common Task Properties
    #
    fieldnumbers=$( pwenv fieldnumbers )
    for path in $TASKS; do
	taskmeta set $path command "./pwfarm_run.sh --field"
	taskmeta set $path prompterr "true"
	taskmeta set $path sudo "false"
	taskmeta set $path outputdir "$( stored_run_path_local $OWNER $RUNID $(taskmeta get $path nid) )"
	if [ $(len $TASKS) -gt $(len $fieldnumbers) ]; then
	    taskmeta set $path statusid "$(taskmeta get $path id)"
	fi
    done

    ###
    ### Create Payload
    ###

    cp $0 $PAYLOAD_DIR/pwfarm_run.sh || exit 1

    function cpopt()
    {
	if [ "$1" != "nil" ]; then
	    cp "$1" "$PAYLOAD_DIR/$2" || exit 1
	fi
    }

    cpopt "$WORLDFILE" worldfile
    cpopt "$OVERLAY" parms.wfo
    cpopt "$PRERUN" prerun.sh
    cpopt "$POSTRUN" postrun.sh
    cpopt "$INPUT_ZIP" input.zip

    echo "$OWNER" > $PAYLOAD_DIR/owner
    echo "$RUNID" > $PAYLOAD_DIR/runid

    for x in "$FETCH_LIST"; do
	echo "$x" >> $RUN_PACKAGE_DIR/input
    done

    if [ "$WORLDFILE" == "nil" ]; then
	#
	# Package the Checksums
	#
	for path_task in $TASKS; do
	    rundir=$(taskmeta get $path_task outputdir)
	    checksums=$RUN_PACKAGE_DIR/checksums_$( taskmeta get $path_task nid )

	    $POLYWORLD_SCRIPTS_DIR/archive_delta.sh checksums \
		$checksums \
		$rundir \
		"$FETCH_LIST"
	done
    fi

    PAYLOAD=$PAYLOAD_DIR/payload.zip

    pushd_quiet .
    cd $PAYLOAD_DIR
    zip -qr $PAYLOAD .
    popd_quiet

    ##
    ## Execute
    ##
    dispatcher dispatch $PAYLOAD $TASKS

    rm -rf $TMP_DIR
    rm -rf $overlay_tmpdir
else
    ##############################
    ###                        ###
    ### EXECUTE ON REMOTE HOST ###
    ###                        ###
    ##############################

    if [ ! -e "$POLYWORLD_PWFARM_APP_DIR" ]; then
	err "No app directory! Please do a pwfarm_build."
    fi

    lock_app || exit 1

    PAYLOAD_DIR=$( pwd )
    cd "$POLYWORLD_PWFARM_APP_DIR" || exit 1

    export POLYWORLD_PWFARM_WORLDFILE="$PAYLOAD_DIR/worldfile"
    export POLYWORLD_PWFARM_RUN_PACKAGE=$PAYLOAD_DIR/run_package/input

    export DISPLAY=:0.0 # for Linux -- allow graphics from ssh
    ulimit -n 4096      # for Mac -- allow enough file descriptors
    ulimit -c unlimited # for Mac -- generate core dump on trap
    export PATH=$( canonpath scripts ):$( canonpath bin ):$PATH

    OWNER=$(cat $PAYLOAD_DIR/owner)
    RUNID=$(cat $PAYLOAD_DIR/runid)
    NID=$(PWFARM_TASKMETA get nid)
    BATCHID=$( PWFARM_TASKMETA get batchid )

    ###
    ### If a run is already here, store it away.
    ###
    store_orphan_run ./run

    # These files will be used for run identity if our run gets orphaned.
    echo $OWNER > ./owner
    echo $RUNID > ./runid
    echo $NID > ./nid

    ###
    ### If we're running Polyworld, make sure a run with a conflicting ID doesn't already exist.
    ###
    if [ -e "$POLYWORLD_PWFARM_WORLDFILE" ]; then
	if conflicting_run_exists $OWNER $RUNID $NID $BATCHID; then
	    # conflicting_run_exists will print details to stderr
	    err "Aborting due to conflicting run!"
	fi
    fi

    ###
    ### Unpack the input zip
    ###
    if [ -e $PAYLOAD_DIR/input.zip ]; then
	export POLYWORLD_PWFARM_INPUT="$PAYLOAD_DIR/input"

	rm -rf $POLYWORLD_PWFARM_INPUT
	mkdir -p $POLYWORLD_PWFARM_INPUT || exit 1
	unzip $PAYLOAD_DIR/input.zip -d $POLYWORLD_PWFARM_INPUT || exit 1
    fi

    ###
    ### Execute the prerun script if it exists
    ###
    if [ -f $PAYLOAD_DIR/prerun.sh ]; then
	PWFARM_STATUS "Config Script"
	chmod +x $PAYLOAD_DIR/prerun.sh
	$PAYLOAD_DIR/prerun.sh || exit 1
    fi


    ###
    ### If no worldfile, then relocate requested run to current directory
    ###
    if [ ! -e $POLYWORLD_PWFARM_WORLDFILE ]; then
	unstore_run $OWNER $RUNID $NID ./run || exit 1
    fi

    ###
    ### Execute Polyworld if worldfile exists
    ###
    if [ -e $POLYWORLD_PWFARM_WORLDFILE ]; then
	###
	### Process Parms Overlay
	###
	if [ -e "$PAYLOAD_DIR/parms.wfo" ]; then
	    proputil overlay $POLYWORLD_PWFARM_WORLDFILE "$PAYLOAD_DIR/parms.wfo" "$NID" > ./worldfile || exit 1
	else
	    ###
	    ### Set the seed based on NID
	    ###
	    if [ "$NID" == "0" ]; then
		seed=42
	    else
		seed="$NID"
	    fi
	    cp $POLYWORLD_PWFARM_WORLDFILE ./worldfile || exit 1
	    ./scripts/wfutil edit ./worldfile InitSeed=$seed || exit 1
	fi


	########################
	###                  ###
	### Run Polyworld!!! ###
	###                  ###
	########################
	PWFARM_STATUS "Polyworld"

	./Polyworld --status ./worldfile
	exitval=$?

	if ! is_good_run --exit $exitval ./run; then
	    store_failed_run $OWNER $RUNID $NID ./run
	    exit 1
	fi

	cp $POLYWORLD_PWFARM_WORLDFILE run/farm.wf
	if [ -e "$PAYLOAD_DIR/parms.wfo" ]; then
	    cp "$PAYLOAD_DIR/parms.wfo" run/
	fi

	mkdir -p run/.pwfarm
	echo $( pwenv fieldnumber ) > run/.pwfarm/fieldnumber
	echo $( PWFARM_TASKMETA get nid ) > run/.pwfarm/nid
	echo $BATCHID > run/.pwfarm/batchid
    fi

    ###
    ### Execute the postrun script
    ###
    postrun_failed=false

    if [ -f $PAYLOAD_DIR/postrun.sh ]; then
	PWFARM_STATUS "Analysis Script"

	chmod +x $PAYLOAD_DIR/postrun.sh

	if ! $PAYLOAD_DIR/postrun.sh; then
	    postrun_failed=true
	fi
    fi

    PWFARM_STATUS "Package Run"

    ###
    ### Store a code snapshot
    ###
    if [ -e $PAYLOAD_DIR/postrun.sh ]; then
	scripts/run_history.sh src run $postrun_failed $PAYLOAD_DIR/postrun.sh
    else
	scripts/run_history.sh src run "false"
    fi

    ###
    ### Create archive pulled to workstation
    ###
    cd run || err "postrun script moved ./run!!!"

    $POLYWORLD_PWFARM_SCRIPTS_DIR/archive_delta.sh archive \
	-e $PAYLOAD_DIR/run_package/checksums_$NID \
	$PWFARM_OUTPUT_FILE \
	. \
	"$( cat $POLYWORLD_PWFARM_RUN_PACKAGE ) .pwfarm/*"

    cd ..

    ###
    ### Store run
    ###
    store_good_run $OWNER $RUNID $NID "./run"

    ###
    ### Exit
    ###
    unlock_app

    if $postrun_failed; then
	exit 1
    else
	exit 0
    fi
fi