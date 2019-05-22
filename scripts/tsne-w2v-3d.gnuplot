set loadpath './scripts'
load 'styles.gnuplot'

set terminal svg enhanced size 1000,800 background rgb '#222222'
set output 'tsne-w2v-3d.svg'

set datafile separator ","
splot 'target/tsne-w2v-3d.csv' using 1:2:3:4 with labels font "system-ui,9" textcolor "#FCAB9D" notitle