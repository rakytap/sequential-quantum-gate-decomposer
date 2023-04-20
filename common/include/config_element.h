/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file config_element.h
    \brief Header file for a class describing a universal parameter with name and value stored in a config map.
*/

#ifndef CONFIG_ELEMENT_H
#define CONFIG_ELEMENT_H


#include <string>


/**
@brief A class describing a universal configuration element. Can store one element at once, by setting a value other attributes are reset.
*/
class Config_Element  {



protected:

    /// The name of the configuration property
    std::string name;

    /// variable to store double parameter value
    double dval;
    /// variable to store bool parameter value
    bool bval;
    /// variable to store int parameter value
    int ival;
    /// variable to store long long parameter value
    long long llval;
    /// variable to store unsigned long long parameter value
    unsigned long long ullval;

public:

/** Nullary constructor of the class
@return An instance of the class
*/
Config_Element();

/**
@brief Destructor of the class
*/
virtual ~Config_Element();


/**
@brief Call to set a double value
@param name The name of the property
@param val_ The value of the property
*/
void set_property( std::string name_, double val_ );


/**
@brief Call to set a bool value
@param name The name of the property
@param val_ The value of the property
*/
void set_property( std::string name_, bool val_ );


/**
@brief Call to set an integer value
@param name The name of the property
@param val_ The value of the property
*/
void set_property( std::string name_, int val_ );


/**
@brief Call to set a long long value
@param name The name of the property
@param val_ The value of the property
*/
void set_property( std::string name_, long long val_ );


/**
@brief Call to set an unsigned long long value
@param name The name of the property
@param val_ The value of the property
*/
void set_property( std::string name_, unsigned long long val_ );





/**
@brief Call to get a double value
@param name The name of the property
@param val_ The value of the property
*/
void get_property( double& val_ );


/**
@brief Call to get a bool value
@param name The name of the property
@param val_ The value of the property
*/
void get_property( bool& val_ );


/**
@brief Call to get an integer value
@param name The name of the property
@param val_ The value of the property
*/
void get_property( int& val_ );


/**
@brief Call to get a long long value
@param name The name of the property
@param val_ The value of the property
*/
void get_property( long long& val_ );


/**
@brief Call to get an unsigned long long value
@param name The name of the property
@param val_ The value of the property
*/
void get_property( unsigned long long& val_ );




};


#endif //Config_element
