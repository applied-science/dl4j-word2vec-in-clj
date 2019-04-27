set datafile separator ","
set terminal png
set output 'tsne-plot-3d.png'
splot 'target/tsne-standard-coords-3d.csv' using 1:2:3:4 with labels font "Times,8"
