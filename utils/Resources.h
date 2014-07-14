#pragma once

#include <fstream>
#include <string>

namespace proplib { class Document; class SchemaDocument; }

class Resources
{
 public:
    static void init();

    static std::string get_user_path(std::string path);
    static std::string get_pw_path(std::string path);

	static std::string get_polygons_path(std::string name);

 private:
	static std::string find( std::string name );
};
