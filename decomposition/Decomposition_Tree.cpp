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
/*! \file Decomposition_Base.cpp
    \brief Class containing basic methods for the decomposition process.
*/


#include "Decomposition_Tree.h"
#include <limits>




/** Nullary constructor of the class
@return An instance of the class
*/
Decomposition_Tree_Node::Decomposition_Tree_Node() {

  /// logical value indicating whether the decomposition tree node is active or not
  active = true;

  // The current level in the decomposition tree
  level = 0;

  // The strored decomposition layer
  layer = NULL;

  // number of active children nodes
  active_children = 0;

  // the obtained cost function
  cost_func_val = std::numeric_limits<double>::max();

  //the array of parameters for which the cost function value was obtained
  optimized_parameters = Matrix_real(0,0);

  // the child node in the decomposition tree with the minimal cost function
  minimal_child = NULL;

  // The parent node in the decomposition tree
  parent = NULL;

  // mutual exclusion to count the references for class instances referring to the same data.
  tree_mutex = new tbb::spin_mutex();

}




/** 
@ brief Destructor of the class
*/
Decomposition_Tree_Node::~Decomposition_Tree_Node() {

    // release children nodes
    for ( tbb::concurrent_vector<Decomposition_Tree_Node*>::iterator it = children.begin(); it != children.end(); it++) {
        if ( *it != NULL ) {
            delete( *it );
        }

    }

    minimal_child = NULL;

    if ( tree_mutex !=NULL) {
        tree_mutex->~spin_mutex();
        delete tree_mutex;
        tree_mutex=NULL;
    }

}






/**
@brief Add a child node to the decomposition tree and update the minimal node if necessary
*/
void 
Decomposition_Tree_Node::add_Child( Decomposition_Tree_Node* child ) {

    child->parent = this;

    // update the level of the child node
    child->level = level + 1;


    
    // update the minimal node
    {
        tbb::spin_mutex::scoped_lock my_lock{*tree_mutex};

    // add child to the children list
    children.push_back( child );
    
        if ( minimal_child == NULL ) {
            minimal_child = child;
        }
        else if ( minimal_child->cost_func_val > child->cost_func_val ) {
            minimal_child = child;
        }

        active_children++;
//std::cout << "adding child: " << active_children << std::endl;

    }

}






/**
@brief Add a child node to the decomposition tree and update the minimal node if necessary
*/
void 
Decomposition_Tree_Node::deactivate_Child( Decomposition_Tree_Node* child ) {

    double minimal_cost_func_val = std::numeric_limits<double>::max();

    {
        minimal_child = NULL;

        tbb::spin_mutex::scoped_lock my_lock{*tree_mutex};

        // delete a child node and find the new minimal node
        for ( tbb::concurrent_vector<Decomposition_Tree_Node*>::iterator it = children.begin(); it != children.end(); it++) {

            // set the cost value of the child node to highest possible value and clear the optimized parameters
            if ( *it == child ) {
                delete( child );
                *it = NULL;
                active_children--;
/*
                child->cost_func_val = std::numeric_limits<double>::max();
                child->optimized_parameters = Matrix_real(0,0);
                child->active = false;
//std::cout << "deactivating child: " << active_children  << std::endl;
*/
                continue;
            }


            if ( (*it) != NULL &&  (minimal_cost_func_val > (*it)->cost_func_val) ) {
                minimal_child = *it;
                minimal_cost_func_val = (*it)->cost_func_val;
            }

        }

    }

}




/**
@brief ????????????
*/
void 
Decomposition_Tree_Node::print_active_children() {

    int level_loc = level;
    std::cout << "active children from level: "<< level_loc <<std::endl;
    Decomposition_Tree_Node* parent_loc = parent;
    while ( level_loc > 0 ) {
        //std::cout << "parent: " << parent << std::endl;
        if ( parent_loc == NULL ) {
            std::cout << "oooooooooooooo" << std::endl;
            break;
        }
        std::cout  << parent_loc->active_children<< ", ";
        level_loc--;// = parent->level;
        parent_loc = parent_loc->parent;
    }

    std::cout << std::endl;



}




