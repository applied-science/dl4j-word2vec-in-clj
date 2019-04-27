# Plot Data with gnuplot
# !!! Possible error: plot was recently deprecated.
set terminal png
set datafile separator ","
set terminal png
set output 'tsne-plot.png'
plot 'target/tsne-standard-coords.csv' using 1:2:3 with labels font "Times,8"
