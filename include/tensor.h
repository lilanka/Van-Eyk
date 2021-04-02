#include <vector>

using namespace std;

template <typename dtype>
class Tensor {
  public:
    /* create 1-D tensor */ 
    Tensor(int id1) : dim(1) {};
    /* create 2-D tensor */
    Tensor(int id1, int id2) : dim(2) {}; 
    /* create 3-D tensor */
    Tensor(int id1, int id2, int id3) : dim(3) {}; 
    /* create 4-D tensor */
    Tensor(int id1, int id2, int id3, int id4) : dim(4) {}; 
    /* create 5-D tensor */
    Tensor(int id1, int id2, int id3, int id4, int id5) : dim(5) {}; 
    /* create 6-D tensor */
    Tensor(int id1, int id2, int id3, int id4, int id5, int id6) : dim(6) {}; 

    /* Creating custom tensor */
    //vector< test() {return id1;} 
  
  private:
    int dim;
    vector<int> ids;

};
