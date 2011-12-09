if [ -z "$BASH_SOURCE" ]; then
    echo "polyworld/scripts/farm/__lib.sh: Require \$BASH_SOURCE." >&2
    echo "Please use a version of bash >= 3" >&2
    exit 1
fi

source $( dirname $BASH_SOURCE )/__pwfarm_config.sh


function canonpath()
{
    python -c "import os.path; print os.path.realpath('$1')"
}

function canondirname()
{
    dirname `canonpath "$1"`
}

export PWFARM_SCRIPTS_DIR=$( canondirname $BASH_SOURCE )
export POLYWORLD_DIR=$( canonpath $PWFARM_SCRIPTS_DIR/../.. )

function pushd_quiet()
{
    pushd $* > /dev/null
}

function popd_quiet()
{
    popd $* > /dev/null
}

function err()
{
    echo "$( basename $0 ):" "$*">&2
    exit 1
}

function require()
{
    if [ -z "$1" ]; then
	shift
	err "Missing required parameter: $*"
    fi
}

function repeat_til_success
{
    if [ "$1" == "--display" ]; then
	local display="$2"
	shift 2
    else
	local display="$*"
    fi
    local errtime="0"
    local errmsg="/tmp/pwfarm.repeat.err.txt.$$"

    while ! $* 2>$errmsg; do
	local now=$( date '+%s' )

	# Show error message if more than 20 seconds has elapsed since last error
	if [ $(( $now - $errtime )) -gt 20 ]; then
	    echo >&2
	    echo "FAILED: $display" >&2
	    echo "ERR: $( cat $errmsg )" >&2
	    echo "  Repeating until success..." >&2
	fi

	errtime=$now
	# give user chance to ctrl-c
	sleep 5
    done

    rm -f $errmsg

    if [ "$errtime" != "0" ]; then
	echo "RECOVERED FROM ERROR: $display"
    fi
}


function is_process_alive()
{
    local pid_file="$1"
    require "$pid_file" "is_process_alive pid_file arg"

    if [ ! -e "$pid_file" ]; then
	return 1
    fi

    local pid=$( cat $pid_file )
    if [ -z $pid ]; then
	return 1
    fi

    ps -p $pid > /dev/null
}

function mutex_trylock()
{
    local timeout=1
    if [ "$1" == "--timeout" ]; then
	require "$2" "mutex_trylock --timeout arg"
	shift
	timeout="$1"
	shift
    fi

    local mutex_name=$1
    require "$mutex_name" "mutex_lock( mutex_name )"

    while ! mkdir $mutex_name 2>/dev/null; do
	timeout=$(( timeout - 1 ))
	if [ $timeout == 0 ]; then
	    return 1
	fi
	sleep 1
    done

    return 0
}

function mutex_lock()
{
    local mutex_name=$1
    require "$mutex_name" "mutex_lock( mutex_name )"

    local blocked="false"
    while ! mkdir $mutex_name 2>/dev/null; do
	if ! $blocked; then
	    echo "blocking on mutex $mutex_name..."
	    blocked="true"
	fi
	sleep 1
    done

    if $blocked; then
	echo "successfully locked mutex $mutex_name"
    fi
}

function mutex_unlock()
{
    local mutex_name=$1
    require "$mutex_name" "mutex_unlock( mutex_name )"

    if [ ! -e "$mutex_name" ]; then
	err "mutex_unlock invoked on non-locked mutex $mutex_name"
    fi

    rmdir "$mutex_name"
}

function pwquery()
{
    __pwfarm_config query $*
}

function pwenv()
{
    __pwfarm_config env get $*
}

function fieldhostname_from_num()
{
    local id=$( printf "%02d" $1 2>/dev/null ) || err "Invalid field number: $1"
    local hostname=pw$id
    echo $hostname
}

function fieldhostnames_from_nums()
{
    for x in $*; do 
	fieldhostname_from_num $x
    done
}

function fieldhost_from_name()
{
    eval pwhost=\$$1
    echo $pwhost
}

function ensure_farm_session()
{
    if [ -z $( pwenv farmname ) ]; then
	err "You haven't set your farm context."
    fi
    if [ -z $( pwenv sessionname ) ]; then
	err "You haven't set your session context."
    fi

}

function ls_color_opt()
{
    case $( uname ) in
	Darwin)
	    echo -G
	    ;;
	Linux)
	    echo --color
	    ;;
    esac
}

function to_lower() 
{
    echo $1 | tr "[:upper:]" "[:lower:]"
}

function to_upper()
{
    echo $1 | tr  "[:lower:]" "[:upper:]"
}

function is_empty_directory()
{
    local path="$1"

    [ -d "$path" ] && [ -z $( ls -A "$path" ) ];
}

function trim()
{
    local val="$1"
    if [ "$val" == "-" ]; then
	read  -rd '' val
    else
	read  -rd '' val <<< "$val"
    fi
    echo "$val"
}

function show_file_gui()
{
    case `uname` in
	Linux)
	    gnome-open $1 || kde-open $1
	    ;;
	Darwin)
	    open $1
	    ;;
	*)
	    err "Unknown Operating System (`uname`)"
	    ;;
    esac
}

function encode_args()
{
    while [ $# != 0 ]; do
	echo -n \'$1\'" "
	shift
    done
}

function decode_args()
{
    local result=""

    while [ $# != 0 ]; do
	local decoded=$(echo -n $1 | sed -e "s/^'//" -e "s/'$//")
	result="$result $decoded"
	shift
    done

    trim "$result"
}