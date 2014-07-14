#include "Resources.h"

#include <string.h>
#include <unistd.h>

#include "error.h"
#include "misc.h"
#include "proplib.h"

using namespace std;

static const char *RPATH[] = {"./Polyworld.app/Contents/Resources/",
							  "./objects/",
							  "./",
							  NULL};

static string home;
static string pwd;

//===========================================================================
// Resources
//===========================================================================

void Resources::init() {
	// Make sure we're in an appropriate working directory
#if __linux__
	{
		char exe[1024];
		int rc = readlink( "/proc/self/exe", exe, sizeof(exe) );
		exe[rc] = 0;
		char *lastslash = strrchr( exe, '/' );
		*lastslash = 0;

        home = exe;
        pwd = getenv("PWD");
	}
#else
    home = ".";
    pwd = ".";
#endif
}

static string pathcat(string cd, string path) {
    if(path[0] == '/')
        return path;
    else {
        if(path[0] == '.' && path[1] == '/')
            path = path.substr(2);
        return cd + "/" + path;
    }
}

string Resources::get_user_path(string path) {
    return pathcat(pwd, path);
}

string Resources::get_pw_path(string path) {
    return pathcat(home, path);
}


//---------------------------------------------------------------------------
// Resources::get_polygons_path()
//---------------------------------------------------------------------------

string Resources::get_polygons_path(string name)
{
	return find(name + ".obj");
}

//---------------------------------------------------------------------------
// Resources::find()
//---------------------------------------------------------------------------

string Resources::find( string name )
{
	for( const char **path = RPATH;
		 *path;
		 path++ )
	{
        string fullpath = get_pw_path( string(*path) + name );

		if( exists(fullpath) )
		{
			return fullpath;
		}
	}

	return "";
}
