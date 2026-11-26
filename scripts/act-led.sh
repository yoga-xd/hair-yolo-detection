# /usr/local/bin/act-led：Pi5的ACT灯为低电平有效（on=写0，off=写255）
case "$1" in
  on)  echo none > /sys/class/leds/ACT/trigger; echo 0   > /sys/class/leds/ACT/brightness ;;
  off) echo none > /sys/class/leds/ACT/trigger; echo 255 > /sys/class/leds/ACT/brightness ;;
  restore) echo mmc0 > /sys/class/leds/ACT/trigger ;;
esac
