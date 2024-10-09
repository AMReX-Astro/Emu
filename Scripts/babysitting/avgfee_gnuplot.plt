set term pdf
set output 'avgfee.pdf'
set key autotitle columnhead
set xlabel 'time (s)'
set ylabel 'N00 (cm^-3)'
set yrange [0:*]
plot 'reduced0D.dat' u 2:5 w l