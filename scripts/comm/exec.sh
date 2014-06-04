#!/bin/bash -x

trialsdir=$(dirname $0)
batch_size=50
batch_success=$(python -c "print int(${batch_size} * 0.9)")
max_steps=1000

scripts/comm/mkbranch.py --max-steps $max_steps --food-difficulty 0 --random-single-food 0 > $trialsdir/trial0.wf
./Polyworld $trialsdir/trial0.wf
mv run $trialsdir/run0

food_difficulty=0

time_total=0
success_count=0
attempt_count=0
prev_time_avg=0

i=1
while true; do
    attempt_count=$((attempt_count + 1))

    cat <<EOF
========================================
i: $i
attempt: $attempt_count
success: $success_count
prev time avg: $prev_time_avg
time total: $time_total
food difficulty: $food_difficulty
========================================
EOF

    scripts/genomeSeed --fittest $trialsdir/run$((i - 1))
    scripts/comm/mkbranch.py --max-steps $max_steps --food-difficulty $food_difficulty --random-single-food $i --seed-from-run > $trialsdir/trial$i.wf
    ./Polyworld $trialsdir/trial$i.wf

    if grep Eat run/endReason.txt > /dev/null; then
        success_count=$((success_count + 1))
    fi

    t=$(cat run/endStep.txt)
    time_total=$(( time_total + t ))
    cat <<EOF
  t: $t
  time total: $time_total
EOF

    if [ $attempt_count == ${batch_size} ]; then
        time_avg=$(( time_total / ${batch_size} ))
        cat <<EOF
  time_avg: $time_avg
  prev_time_avg: $prev_time_avg
EOF
        if [ $success_count -ge $batch_success ]; then
            if python -c "exit(0 if (abs( float($time_avg - $prev_time_avg) / $time_avg) < 0.05) else 1)"; then
                if [ $food_difficulty == 2 ]; then
                    echo "WE WON!!!"
                else
                    echo "PROMOTING FOOD DIFFICULTY"
                    food_difficulty=$((food_difficulty + 1))
                fi
            fi
        fi

        attempt_count=0
        success_count=0
        prev_time_avg=$time_avg
        time_total=0
    fi

    mv run $trialsdir/run$((i))

    i=$((i + 1))
done 