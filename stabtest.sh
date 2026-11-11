#!/bin/sh
# usage: [RC_*=..] stabtest.sh <label>: how much the irradiance changes under a 3 pixel turn
yaw=$(python -c "import math;print(math.degrees(math.atan(3/ (360/math.tan(math.radians(22.5))))))")
sh shoot.sh t_a irradiance; RC_YAW=$yaw RC_SETTLE=60 sh shoot.sh t_b irradiance
printf '%-28s' "$1"; python stab.py shots/t_a.png shots/t_b.png 6 | sed 's/.*: //'
