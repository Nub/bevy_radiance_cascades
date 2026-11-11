#!/bin/sh
# Renders the room example in a hidden window and saves one frame.
# usage: shoot.sh <name> [off|on|indirect|irradiance]   -> shots/<name>.png
mkdir -p shots
RC_MODE=${2:-on} RC_SHOT=shots/$1.png timeout 120 ./target/debug/examples/room.exe 2>&1 \
    | grep -v gilrs | grep -E 'ERROR|WARN|error|panic' -A 25 | head -24
