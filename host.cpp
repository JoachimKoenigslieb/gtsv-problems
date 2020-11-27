#include <iostream>
#include <string.h>
#include "cnpy.h"
#include <iostream>

#include <random>
#include "xcl2.cpp"

extern "C" {
    void dgtsv_(int*, int*, double*, double*, double*, double*, int*, int*);
}

// Memory alignment
template <typename T>
T *aligned_alloc(std::size_t num)
{
	void *ptr = nullptr;
	if (posix_memalign(&ptr, 4096, num * sizeof(T)))
	{
		throw std::bad_alloc();
	}
	return reinterpret_cast<T *>(ptr);
}

int next_largets_factor_2(int n){
	int factor_2 = 1;  
	while (factor_2 < n){
		factor_2 *= 2;
	}
	return factor_2;
}


double check_tridiag_solution(double* a_tri, double* b_tri, double* c_tri, double* d_tri, double* solution, int N, int verbose){
	double rhs_calculated;
	double error;

	rhs_calculated = b_tri[0] * solution[0] + c_tri[0] * solution[1];
	error += abs(rhs_calculated - d_tri[0]);
	if(verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[0] << " Accumulated error: " << error << std::endl;}

	for (int i=1; i<(N-1); i++){ //we neglect special cases
		rhs_calculated = a_tri[i] * solution[i-1] + b_tri[i] * solution[i] + c_tri[i] * solution[i+1];
		error += abs(rhs_calculated - d_tri[i]);
		if (verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[i] << " Accumulated error: " << error <<std::endl;}
	}

	rhs_calculated = a_tri[N-1] * solution[N-2] + b_tri[N-1] * solution[N-1];
	error += abs( rhs_calculated - d_tri[N-1]);
	if (verbose){std::cout << "calculated rhs: " << rhs_calculated << " reference rhs: " << d_tri[N-1] << " Accumulated error: " << error <<std::endl;}

	return error;
}

void run_gtsv(std::string kernel_name, int kernel_size, std::vector<double *> &inputs, std::vector<cl::Device> &devices, cl::Context &context, cl::Program::Binaries &bins, cl::CommandQueue &q)
{
        // this is a helper function to execute a kernel.
        {
                cl::Program program(context, devices, bins); //Note. we use devices not device here!!!
                cl::Kernel kernel(program, kernel_name.data());

                 // DDR Settings
		std::vector<cl_mem_ext_ptr_t> mext_io(4);
		mext_io[0].flags = XCL_MEM_DDR_BANK0;
		mext_io[1].flags = XCL_MEM_DDR_BANK0;
		mext_io[2].flags = XCL_MEM_DDR_BANK0;
		mext_io[3].flags = XCL_MEM_DDR_BANK0;

		mext_io[0].obj = inputs[0];
		mext_io[0].param = 0;
		mext_io[1].obj = inputs[1];
		mext_io[1].param = 0;
		mext_io[2].obj = inputs[2];
		mext_io[2].param = 0;
		mext_io[3].obj = inputs[3];
		mext_io[3].param = 0;

		// Create device buffer and map dev buf to host buf
		cl::Buffer matdiaglow_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
										sizeof(double) * kernel_size, &mext_io[0]);
		cl::Buffer matdiag_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
									sizeof(double) * kernel_size, &mext_io[1]);
		cl::Buffer matdiagup_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
										sizeof(double) * kernel_size, &mext_io[2]);
		cl::Buffer rhs_buffer = cl::Buffer(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
								sizeof(double) * kernel_size, &mext_io[3]);

		// Data transfer from host buffer to device buffer
		std::vector<std::vector<cl::Event> > kernel_evt(2);
		kernel_evt[0].resize(1);
		kernel_evt[1].resize(1);

		std::vector<cl::Memory> ob_in, ob_out;
		ob_in.push_back(matdiaglow_buffer);
		ob_in.push_back(matdiag_buffer);
		ob_in.push_back(matdiagup_buffer);
		ob_in.push_back(rhs_buffer);

		ob_out.push_back(rhs_buffer);

		q.enqueueMigrateMemObjects(ob_in, 0, nullptr, &kernel_evt[0][0]); // 0 : migrate from host to dev
		q.finish();
		std::cout << "INFO: Finish data transfer from host to device" << std::endl;


		// Setup kernel
		kernel.setArg(0, kernel_size);
		kernel.setArg(1, matdiaglow_buffer);
		kernel.setArg(2, matdiag_buffer);
		kernel.setArg(3, matdiagup_buffer);
		kernel.setArg(4, rhs_buffer);
		q.finish();
		std::cout << "INFO: Finish kernel setup" << std::endl;

		q.enqueueTask(kernel, nullptr, nullptr);
		q.finish();
		q.enqueueMigrateMemObjects(ob_out, 1, nullptr, nullptr); // 1 : migrate from dev to host
		q.finish();


		}
}

int main(){
	std::string xclbin_path = "./kernels.xclbin";
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[0];
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	cl::Program::Binaries bins = xcl::import_binary_file(xclbin_path);
	devices.resize(1);

	int N = 4096;
	//N = next_largets_factor_2(N); // we try to pad to a factor of two in the hopes that this is what causes errors!
	std::cout << N << std::endl;

	double* a_tri = aligned_alloc<double>(N);
	double* b_tri = aligned_alloc<double>(N);
	double* c_tri = aligned_alloc<double>(N);
	double* d_tri = aligned_alloc<double>(N);
	double* rhs_copy = aligned_alloc<double>(N);

	double err = 0.0;

	double lower_bound = 0;
	double upper_bound = 10;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re(4); //fixed seed

	int mode = 0;
	int verbose = 0;

	//random random data into tris and rhs. Save a copy of rhs, this will be overwritten in-place!
	if (mode == 0) {
		for (int i=0; i<N; i++){
			a_tri[i] = unif(re);
			b_tri[i] = unif(re);
			c_tri[i] = unif(re);
			d_tri[i] = unif(re);
			rhs_copy[i] = d_tri[i]; //save this, as d_tri will get written to in-place!
		}
	} else {
		for (int i = 0; i < N; i++) {
			a_tri[i] = -1.0;
			b_tri[i] = 2.0;
			c_tri[i] = -1.0;
			d_tri[i] = 0.0;
			rhs_copy[i] = d_tri[i];
		};

		d_tri[N - 1] = 1.0;
		d_tri[0] = 1.0;
		rhs_copy[0] = d_tri[0];
		rhs_copy[N -1] = d_tri[N -1];
	}

	a_tri[0] = 0.0;
	c_tri[N-1] = 0.0;
	
	std::vector<double*> inputs, outputs;

	inputs = {a_tri, b_tri, c_tri, d_tri};
	run_gtsv("gtsv", N, inputs, devices, context, bins, q); //this outputs solution into d_tri 
	
	double xilinx_error;

	xilinx_error = check_tridiag_solution(a_tri, b_tri, c_tri, rhs_copy, d_tri, N, 0);

	double sum_solution = 0;
	for (int i=0; i<N; i++){
		sum_solution += d_tri[i];
	}
	std::cout << "accumulated error of solution /w xf::solver " << xilinx_error << std::endl;
 	std::cout << "sum of solution /w xf::solver: " << sum_solution << std::endl;

	//Now for lapack test:
	std::default_random_engine reMix(4); //fixed seed
	double* a_tri_copy = aligned_alloc<double>(N);
	double* b_tri_copy = aligned_alloc<double>(N);
	double* c_tri_copy = aligned_alloc<double>(N);

	if (mode == 0) {
		for (int i=0; i<N; i++){
			a_tri[i] = unif(reMix);
			b_tri[i] = unif(reMix);
			c_tri[i] = unif(reMix);
			d_tri[i] = unif(reMix);
			rhs_copy[i] = d_tri[i]; //save this, as d_tri will get written to in-place!
			a_tri_copy[i] = a_tri[i];
			b_tri_copy[i] = b_tri[i];
			c_tri_copy[i] = c_tri[i]; //lapack will destroy my data
		}
	} else {
		for (int i = 0; i < N; i++) {
			a_tri[i] = -1.0;
			b_tri[i] = 2.0;
			c_tri[i] = -1.0;
			d_tri[i] = 0.0;
			rhs_copy[i] = d_tri[i];
			a_tri_copy[i] = a_tri[i];
			b_tri_copy[i] = b_tri[i];
			c_tri_copy[i] = c_tri[i]; //lapack will destroy my data
		};

		d_tri[N - 1] = 1.0;
		d_tri[0] = 1.0;
		rhs_copy[0] = d_tri[0];
		rhs_copy[N -1] = d_tri[N -1];
	}

	a_tri[0] = 0.0;
	c_tri[N-1] = 0.0;

	int info = 0;
	int NHRS = 1;
	int LDB = N; 

	dgtsv_(&N, &NHRS, &a_tri[1], b_tri, c_tri, d_tri, &LDB, &info); //call lapack

	double lapack_error;

	lapack_error = check_tridiag_solution(a_tri_copy, b_tri_copy, c_tri_copy, rhs_copy, d_tri, N, 0);

	sum_solution = 0;
	for (int i=0; i<N; i++){
		sum_solution += d_tri[i];
	}

	std::cout << "accumulated error of solution /w lapack " << lapack_error << std::endl;
	std::cout << "sum of solution /w lapack: " << sum_solution << std::endl;
 
	return 0;
}