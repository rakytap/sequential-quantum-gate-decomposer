#
#  Copyright 2021 Budapest Quantum Computing Group
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Check for the presence of AVX and figure out the flags to use for it.
macro(CHECK_FOR_AVX)

    include(CheckCXXSourceRuns)
    set(CMAKE_REQUIRED_FLAGS)
    
    # Check AVX
    # Identify the compiler type and set compiler specific options
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # using Clang

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # using GCC
      set(CMAKE_REQUIRED_FLAGS "-mavx")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      # using Intel C++
      set(CMAKE_REQUIRED_FLAGS "-mavx")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # using Visual Studio C++
      set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
    endif()



    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m256d a, b, c;
          const double src[8] = { 1.0D, 2.0D, 3.0D, 4.0D };
          double dst[8];
          a = _mm256_loadu_pd( src );
          b = _mm256_loadu_pd( src );
          c = _mm256_add_pd( a, b );
          _mm256_storeu_pd( dst, c );

          for( int i = 0; i < 4; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }

          return 0;
        }"
        HAVE_AVX_EXTENSIONS)




    # Check AVX2
    # Identify the compiler type and set compiler specific options
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # using Clang

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # using GCC
      set(CMAKE_REQUIRED_FLAGS "-mavx2")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      # using Intel C++
      set(CMAKE_REQUIRED_FLAGS "-mavx2")

    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # using Visual Studio C++
      set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    endif()

    check_cxx_source_runs("
        #include <immintrin.h>
        int main()
        {
          __m256i a, b, c;
          const int src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
          int dst[8];
          a =  _mm256_loadu_si256( (__m256i*)src );
          b =  _mm256_loadu_si256( (__m256i*)src );
          c = _mm256_add_epi32( a, b );
          _mm256_storeu_si256( (__m256i*)dst, c );

          for( int i = 0; i < 8; i++ ){
            if( ( src[i] + src[i] ) != dst[i] ){
              return -1;
            }
          }

          return 0;
        }"
        HAVE_AVX2_EXTENSIONS)


    if (DEFINED ENV{USE_AVX512})
        # Check AVX512F instruction set
        # Identify the compiler type and set compiler specific options
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
          # using Clang

        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
          # using GCC
          set(CMAKE_REQUIRED_FLAGS "-mavx512f")

        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
          # using Intel C++
          set(CMAKE_REQUIRED_FLAGS "-mavx512f")

        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
          # using Visual Studio C++
          set(CMAKE_REQUIRED_FLAGS "/arch:AVX512F")
        endif()

        check_cxx_source_runs("
            #include <immintrin.h>
            int main()
            {

              __m512d a, b, c;
              const double src[8] = { 1.0D, 2.0D, 3.0D, 4.0D, 1.0D, 2.0D, 3.0D, 4.0D };
              double dst[8];
              a = _mm512_loadu_pd( src );
              b = _mm512_loadu_pd( src );
              c = _mm512_add_pd( a, b );
              _mm512_storeu_pd( dst, c );

              for( int i = 0; i < 8; i++ ){
                if( ( src[i] + src[i] ) != dst[i] ){
                  return -1;
                }
              }

              return 0;
            }"
            HAVE_AVX512F_EXTENSIONS)
    endif()





    
endmacro(CHECK_FOR_AVX)
