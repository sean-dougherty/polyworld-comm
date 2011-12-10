#!/bin/bash

source $( dirname $BASH_SOURCE )/__lib.sh || exit 1

FIELD_NUMBER="$1"
require "$FIELD_NUMBER" "farmer field number"
. $PWFARM_SCRIPTS_DIR/pwfarm_config.sh env set fieldnumber $FIELD_NUMBER

FIELD_HOSTNAME=$( fieldhostname_from_num $FIELD_NUMBER )
FIELD_HOST=$( fieldhost_from_name $FIELD_HOSTNAME )
FIELD_SCREEN_SESSION="$( pwenv fieldscreensession )"

FARMER_STATE_DIR=$( pwenv farmerstate_dir ) || exit 1
require "$FARMER_STATE_DIR" "farmer dir cannot be empty!!!"
mkdir -p $FARMER_STATE_DIR
cd $FARMER_STATE_DIR

FIELD_STATE_DIR="$( pwenv fieldstate_dir )" || exit 1
require "$FIELD_STATE_DIR" "field dir cannot be empty!!!"

OSUSER=$( pwenv osuser )

STEPS=$FARMER_STATE_DIR/steps
PID=$FARMER_STATE_DIR/pid

# macro that specifies user, host and a server timeout
SSH="ssh -l $OSUSER -o ServerAliveInterval=30 $FIELD_HOST"

mode="$2"
require "$mode" "farmer mode"


case "$mode" in
    "dispatch")
	# no-op... fall through
	;;
    "recover")
	# no-op... fall through
	;;
    "clear")	
	$SSH "
          if [ -e $FIELD_STATE_DIR/pid ]; then
            kill \$( cat $FIELD_STATE_DIR/pid ) 2>/dev/null ;
          fi ;
          if [ -e $FIELD_STATE_DIR/pid_command ]; then
            kill \$( cat $FIELD_STATE_DIR/pid_command ) 2>/dev/null;
          fi ;
          while screen -ls | grep \"$FIELD_SCREEN_SESSION\" > /dev/null; do
            screen -S \"$FIELD_SCREEN_SESSION\" -X quit ;
            sleep 1 ;
          done ;
          rm -rf $FIELD_STATE_DIR"
	if is_process_alive $PID; then
	    kill $( cat $PID )
	fi
	rm -rf $FARMER_STATE_DIR

	exit 0
	;;
    "disconnect")
	if is_process_alive $PID; then
	    kill $( cat $PID )
	fi
	exit 0
	;;
    "exit")
	if [ -e $FARMER_STATE_DIR/result/exitval ]; then
	    exit $( cat $FARMER_STATE_DIR/result/exitval )
	else
	    exit 1
	fi
	;;
    *)
	err "Invalid farmer mode ($mode)"
	;;
esac

################################################################################
#####
##### DISPATCH / RECOVER
#####
################################################################################

BLOB="$3"
require "$BLOB" "blob arg"
PASSWORD="$4"
require "$PASSWORD" "password arg"
PROMPT_ERR="$5"
require "$PROMPT_ERR" "prompt_err arg"
COMMAND="$6"
require "$COMMAND" "command arg"
OUTPUT_BASENAME="$7"
OUTPUT_DIR="$8"

if is_process_alive $PID; then
    kill $( cat $PID )
fi
echo $$ > $PID

touch $STEPS

#
# Util functions for maintaining steps file
#
function step_begin()
{
    __step=$1

    ! grep "^$__step$" "$STEPS" 1>/dev/null
}

function step_done()
{
    echo "$__step" >> "$STEPS"
}

#
# Init field state dir
#
if step_begin "init_statedir"; then
    repeat_til_success $SSH "
      if [ -e $BLOB ]; then
        rm -rf $FIELD_STATE_DIR ;
        mkdir -p $FIELD_STATE_DIR ;
        cd $FIELD_STATE_DIR ;
        unzip -q $BLOB ;
        cd payload ;
        unzip -q payload.zip ;
        rm payload.zip ;

        rm $BLOB ;
      fi ;
    "

    step_done
fi

#
# Run script
#
if step_begin "run_script"; then
    
    # set --display so we don't echo password to console
    repeat_til_success \
	--display "ssh __pwfarm_field.sh launch" \
	$SSH -t "
          export PASSWORD=\"$PASSWORD\" ;
          export PROMPT_ERR=\"$PROMPT_ERR\" ;
          $( $PWFARM_SCRIPTS_DIR/pwfarm_config.sh env export )
          cd $FIELD_STATE_DIR ;
          scripts/__pwfarm_field.sh launch
                                    \"$COMMAND\"
    "

    step_done
fi

#
# Fetch result file
#
if step_begin "fetch_result"; then
    repeat_til_success scp $OSUSER@$FIELD_HOST:$FIELD_STATE_DIR/result.zip $FARMER_STATE_DIR

    step_done
fi

#
# Unpack result file
#
if step_begin "unpack_result"; then
    if [ -e result.zip ]; then
	rm -rf result
	mkdir result
	unzip -q -d result result.zip
	rm result.zip
    fi

    step_done
fi

#
# Save log file
#
if step_begin "save_log"; then
    if [ -e result/log ]; then
	cp result/log /tmp/log_field${FIELD_NUMBER}_session$(pwenv sessionname)_farm$(pwenv farmname)
    fi

    step_done
fi

#
# Unpack output file
#
if step_begin "unpack_output"; then
    if [ -e result/output.zip ]; then
	unzipdir="${OUTPUT_DIR}/${OUTPUT_BASENAME}_${FIELD_NUMBER}"
	mkdir -p $unzipdir
	unzip -oq -d $unzipdir result/output.zip
	rm result/output.zip
    fi

    step_done
fi

rm $PID

exit