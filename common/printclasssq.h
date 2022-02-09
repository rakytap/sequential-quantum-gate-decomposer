/*brief Header file for a print class in order to control the verbosity levels of messages. 
*/


#ifndef PRINTCLASSSQ_H
#define PRINTCLASSSQ_H

#include <string>
#include <iostream>
#include <sstream> 
#include <fstream>




class logging 
{ 


public:

 	
 
 	void printnewsq(std::stringstream& ssq, int verbose_level=1); //default verbose_level=1

};

#endif
