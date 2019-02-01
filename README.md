# Parallel-hybrid-matrix-multiplication

Proyecto de multiprocesadores realizado por:
  >Óscar López Montero y
  >Jonathan Moreno Jiménez

Realizamos la multiplicación de matrices de la siguiente forma:
  -Dividimos primero en procesos con MPI
  -Dividimos el trabajo de cada proceso en hilos con Pthreads.
  -Unimos todo el trabajo independiente e imprimimos el resultado por pantalla.
