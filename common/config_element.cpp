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
/*! \file Adam.cpp
    \brief A class for Adam optimization according to https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
*/

#include "config_element.h"


/** Nullary constructor of the class
@return An instance of the class
*/
Config_Element::Config_Element() {



    /// The name of the configuration property
    name = std::string("");

    /// variable to store double parameter value
    dval = 0.0;
    /// variable to store bool parameter value
    bval = false;
    /// variable to store int parameter value
    ival = 0;
    /// variable to store long long parameter value
    llval = 0;
    /// variable to store unsigned long long parameter value
    ullval = 0;


}

/**
@brief Destructor of the class
*/
Config_Element::~Config_Element() {


}


/**
@brief Call to set a double value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::set_property( std::string name_, double val_ ) {



    /// The name of the configuration property
    name = name_;

    /// variable to store double parameter value
    dval = val_;
    /// variable to store bool parameter value
    bval = false;
    /// variable to store int parameter value
    ival = 0;
    /// variable to store long long parameter value
    llval = 0;
    /// variable to store unsigned long long parameter value
    ullval = 0;

}


/**
@brief Call to set a bool value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::set_property( std::string name_, bool val_ ) {


    /// The name of the configuration property
    name = name_;

    /// variable to store double parameter value
    dval = 0.0;
    /// variable to store bool parameter value
    bval = val_;
    /// variable to store int parameter value
    ival = 0;
    /// variable to store long long parameter value
    llval = 0;
    /// variable to store unsigned long long parameter value
    ullval = 0;

}


/**
@brief Call to set an integer value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::set_property( std::string name_, long val_ ) {

    /// The name of the configuration property
    name = name_;

    /// variable to store double parameter value
    dval = 0.0;
    /// variable to store bool parameter value
    bval = false;
    /// variable to store int parameter value
    ival = val_;
    /// variable to store long long parameter value
    llval = 0;
    /// variable to store unsigned long long parameter value
    ullval = 0;

}


/**
@brief Call to set a long long value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::set_property( std::string name_, long long val_ ) {

    /// The name of the configuration property
    name = name_;

    /// variable to store double parameter value
    dval = 0.0;
    /// variable to store bool parameter value
    bval = false;
    /// variable to store int parameter value
    ival = 0;
    /// variable to store long long parameter value
    llval = val_;
    /// variable to store unsigned long long parameter value
    ullval = 0;

}


/**
@brief Call to set an unsigned long long value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::set_property( std::string name_, unsigned long long val_ ) {


    /// The name of the configuration property
    name = name_;

    /// variable to store double parameter value
    dval = 0.0;
    /// variable to store bool parameter value
    bval = false;
    /// variable to store int parameter value
    ival = 0;
    /// variable to store long long parameter value
    llval = 0;
    /// variable to store unsigned long long parameter value
    ullval = val_;

}





/**
@brief Call to get a double value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::get_property( double& val_ ) {

    val_ = dval;

}



/**
@brief Call to get a bool value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::get_property( bool& val_ ) {

    val_ = bval;

}

/**
@brief Call to get an integer value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::get_property( long& val_ ) {

    val_ = ival;

}


/**
@brief Call to get a long long value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::get_property( long long& val_ ) {

    val_ = llval;

}


/**
@brief Call to get an unsigned long long value
@param name The name of the property
@param val_ The value of the property
*/
void 
Config_Element::get_property( unsigned long long& val_ ) {

    val_ = ullval;

}

