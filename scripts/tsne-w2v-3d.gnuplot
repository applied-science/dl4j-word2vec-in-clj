set datafile separator ","
set terminal png
set output 'tsne-w2v-3d.png'
splot 'target/tsne-w2v-3d.csv' using 1:2:3:4 with labels font "Times,8"
