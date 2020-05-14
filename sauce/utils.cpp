#include <fstream>

bool fileExists(const char* name)
{
	std::ifstream f(name);
	return f.good();
}
