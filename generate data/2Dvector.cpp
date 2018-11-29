//
//  2Dvector.cpp
//  
//
//  Created by menglu liang on 11/28/18.
//

#include <iostream>
#include <vector>

int main()
{
    std::vector <std::vector<int> > vec2D(5, std::vector<int>(4, 1));
    
    for(auto vec : vec2D)
    {
        for(auto x : vec)
            std::cout<<x<<" , ";
        
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    
    for(int i = 0; i < 5; i++)
        for(int j = 0; j < 5; j++)
            vec2D[i][j] = i*j;
    
    for(auto vec : vec2D)
    {
        for(auto x : vec)
            std::cout<<x<<" , ";
        
        std::cout << std::endl;
    }
    
    
    vec2D.push_back(std::vector<int>(4, 11));
    
    std::cout << std::endl;
    
    for(auto vec : vec2D)
    {
        for(auto x : vec)
            std::cout<<x<<" , ";
        
        std::cout << std::endl;
    }
    return 0;
}

//Compile the above code using following command,
//g++ –std=c++11 2dvector.cpp
