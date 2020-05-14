#include <iostream>
#include <map>
#include <string>
#include <list>
#include <fstream>
#include <vector>
#include <iterator>
#include <sys/types.h>
#include <dirent.h>
#include <algorithm>

using namespace std;

 
void read_directory(const string& name, vector<string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

list<string> ReadFile(string path)
{
    string line;
    ifstream myfile(path);
    list<string> mylist;
    if(myfile.is_open())
    {
        while(getline(myfile, line))
        {
            mylist.push_back(line);
        }
        myfile.close();
    }
    return mylist;
}


int main()
{
    vector<string> v;
    read_directory("./data", v);
    //copy(v.begin(), v.end(),ostream_iterator<string>(std::cout, "\n"));




    list<string> mylist;
    vector<string>::iterator ptr;
    for(ptr = v.begin(); ptr < v.end(); ptr++)
    {
        //cout<<*ptr<<endl;
        string path = "data/" + *ptr;
        mylist = ReadFile(path);
        list <string> :: iterator it; 
        for(it= mylist.begin(); it != mylist.end() ; it++)
        {
            cout<<*it <<endl;
        }
    }

    



    // map<string, string> dict;
    // dict.insert(pair<string, string>("xin chao", "cac"));
    // // dict.insert(pair<string, string>("xin1","chao1"));
    // // dict.insert(pair<string, string>("xin2","chao2"));
    // cout<<dict["xin chao"];
    //cout<<sizeof(char);
    // cout<<sizeof(string);
    return 0;
}
