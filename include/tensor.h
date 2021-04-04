#include <vector>

using namespace std;

class Tensor {
  public:

    /* Tensors with proper dimensions */
    Tensor(int id1);
    Tensor(int id1, int id2); 
    Tensor(int id1, int id2, int id3); 
    Tensor(int id1, int id2, int id3, int id4); 
    Tensor(int id1, int id2, int id3, int id4, int id5); 
    Tensor(int id1, int id2, int id3, int id4, int id5, int id6); 

    /* Creating custom tensor */
    auto makingTensor(); 

};
