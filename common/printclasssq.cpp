/*used this class to control the verbosity levels of messages. 

*/


#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "printclasssq.h"






void printsq::printnewsq(std::stringstream& ssq, int verbose_level) 

{

	int verbose_main_level=1; //set the verbose main level

	if (verbose_level<verbose_main_level) { //can be seen on the standart output
        	std::cout << ssq.str() <<'\n';

		
        }

	else if (verbose_level==verbose_main_level) { //write to a "debug.txt" file

		std::ofstream debug_file ("debug.txt");
		debug_file << ssq.str() << '\n'; 
		debug_file.close();
		
        }


	else {
 		std::cout<< "" <<'\n'; //write nothing

	}

}

