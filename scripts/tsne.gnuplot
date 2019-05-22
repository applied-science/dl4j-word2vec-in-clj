# Plot Data with gnuplot
# !!! Possible error: plot was recently deprecated.

set loadpath './scripts'
load 'styles.gnuplot'

set terminal svg enhanced size 1000,800 background rgb '#222222'
set output 'tsne-plot.svg'

set datafile separator ","
plot 'target/tsne-standard-coords.csv' using 1:2:3 with labels font  "system-ui,9" textcolor "#FCAB9D" notitle
