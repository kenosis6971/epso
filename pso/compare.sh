#!/bin/tcsh
python3 pso.py -w 0.729 -c1 1.49445 -c2 1.49445 |& tee ./pso.txt
python3 pso.py -w 0.499382 -c1 1.71425 -c2 0.733686 |& tee ./epso.txt
python3 pso.py -w 0.478759 -c1 1.05137 -c2 0.519747 |& tee ./epso-new.txt
