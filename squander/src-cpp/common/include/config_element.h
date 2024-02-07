/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

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
    long ival;
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
void set_property( std::string name_, long val_ );


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
void get_property( long& val_ );


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
