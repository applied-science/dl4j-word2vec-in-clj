# Plot Data with gnuplot
# !!! Possible error: plot was recently deprecated.
set terminal png
set datafile separator ","
set terminal png
set output 'tsne-w2v-plot.png'
plot 'target/tsne-w2v.csv' using 1:2:3 with labels font "Times,8"
