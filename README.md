# project

Implementation of the paper https://vsitzmann.github.io/siren/ (first principle method)


--------------------------------------------------------------------
Heavy libraries are implemented in C/C++
frontend is done by python
just like other DL frameworks
--------------------------------------------------------------------


TODO: 
  setting up the project -> python c/c++ integration 
      Boost.python doesn't work -> heading to pybind11 :: bybind11 works => Operation done
  
  tensor :: planing on eigen3
  implementing the paper 
  
    1. Weighting mechanism :: Initialization (Page 14) -> uniform distribution
    2. Node mechanism :: activation functions (sine) 

---------------------------------------------------------------------
citation:
---------------------------------------------------------------------
@inproceedings{sitzmann2019siren,
                author = {Sitzmann, Vincent
                          and Martel, Julien N.P.
                          and Bergman, Alexander W.
                          and Lindell, David B.
                          and Wetzstein, Gordon},
                title = {Implicit Neural Representations
                          with Periodic Activation Functions},
                booktitle = {Proc. NeurIPS},
                year={2020}
            }
