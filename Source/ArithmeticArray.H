template<typename T, size_t len>
class ArithmeticArray{
public:
  std::array<T,len> data;

  AMREX_GPU_HOST_DEVICE
  ArithmeticArray(){}

  AMREX_GPU_HOST_DEVICE
  ArithmeticArray(T in){
    for(size_t i=0; i<len; i++) data[i] = in;
  }
  
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  ArithmeticArray(ArithmeticArray<Tin,len> in){
    for(size_t i=0; i<len; i++) data[i] = in[i];
  }

  AMREX_GPU_HOST_DEVICE
  inline size_t size() const{return len;}

  AMREX_GPU_HOST_DEVICE
  const T operator[](const unsigned i) const{
          return data[i];
  }
  
  AMREX_GPU_HOST_DEVICE
  T& operator[](const unsigned i){
          return data[i];
  }
  
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  ArithmeticArray<T,len>& operator=(const ArithmeticArray<Tin,len>& input){
	  for(size_t i=0; i<len; i++) data[i] = input[i];
	  return *this;
  }
  AMREX_GPU_HOST_DEVICE
  ArithmeticArray<T,len>& operator=(const double input){
	  for(size_t i=0; i<len; i++) data[i] = input;
	  return *this;
  }
  AMREX_GPU_HOST_DEVICE
  const ArithmeticArray<T,len> operator*(const double scale) const{
	  ArithmeticArray<T,len> result;
	  for(size_t i=0; i<len; i++) result[i] = data[i] * scale;
	  return result;
  }
  AMREX_GPU_HOST_DEVICE
  const ArithmeticArray<T,len> operator/(const double scale) const{
	  double inv_scale = 1./scale;
	  return operator*(inv_scale);
  }
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  const ArithmeticArray<T,len> operator+(const ArithmeticArray<Tin,len>& input) const{
	  ArithmeticArray<T,len> result;
	  for(size_t i=0; i<len; i++) result[i] = data[i] + input[i];
	  return result;
  }
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  const ArithmeticArray<T,len> operator-(const ArithmeticArray<Tin,len>& input) const{
	  ArithmeticArray<T,len> result;
	  for(size_t i=0; i<len; i++) result[i] = data[i] - input[i];
	  return result;
  }
  AMREX_GPU_HOST_DEVICE
  const ArithmeticArray<T,len> operator-() const{
	  ArithmeticArray<T,len> result;
	  for(size_t i=0; i<len; i++) result[i] = -data[i];
	  return result;
  }
  // don't use tuple + operator because it is not an atomic operation
  // but written this way it uses the atomics
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  void operator+=(const ArithmeticArray<Tin,len>& input){
          for(size_t i=0; i<len; i++) data[i] += input[i];
  }
  AMREX_GPU_HOST_DEVICE
  void operator*=(const double scale){
	  for(size_t i=0; i<len; i++) data[i] *= scale;
  }
  template<typename Tin>
  AMREX_GPU_HOST_DEVICE
  bool operator==(const ArithmeticArray<Tin,len>& input){
	  bool isequal = true;
	  for(size_t i=0; i<len; i++) isequal = isequal && (data[i] == input[i]);
	  return isequal;
  }
};
