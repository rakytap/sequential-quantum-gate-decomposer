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
/*! \file Decomposition_Tree.h
    \brief Header file for a class containing basic methods for the decomposition process.
*/

#ifndef DECOMPOSITION_TREE_H
#define DECOMPOSITION_TREE_H


#include "Gates_block.h"
#include <tbb/cache_aligned_allocator.h>
#include <tbb/concurrent_vector.h>



/**
@brief A class containing basic methods for the decomposition process.
*/
class Decomposition_Tree_Node {


public:
  /// logical value indicating whether the decomposition tree node is active or not
  bool active;
  /// The current level in the decomposition tree
  int level;
  /// number of active children nodes
  int active_children;
  /// The strored decomposition layer
  Gates_block* layer;
  /// the obtained cost function
  double cost_func_val;
  /// the array of parameters for which the cost function value was obtained
  Matrix_real optimized_parameters;
  /// the children nodes in the decomposition tree
  tbb::concurrent_vector<Decomposition_Tree_Node*> children;
  /// the child node in the decomposition tree with the minimal cost function
  Decomposition_Tree_Node* minimal_child;
  /// The parent node in the decomposition tree
  Decomposition_Tree_Node* parent;
  /// mutual exclusion to support some serial data processing.
  tbb::spin_mutex* tree_mutex;


public:

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Decomposition_Tree_Node();


/**
@brief Destructor of the class
*/
~Decomposition_Tree_Node();



/**
@brief Add a child node to the decomposition tree and update the minimal node if necessary
*/
void add_Child( Decomposition_Tree_Node* child );


/**
@brief ??????????
*/
void 
deactivate_Child( Decomposition_Tree_Node* child );


/**
@brief ????????????
*/
void print_active_children();


};




#endif //DECOMPOSITION_TREE
