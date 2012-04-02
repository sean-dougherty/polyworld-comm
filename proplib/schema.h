#pragma once

#include "dom.h"

namespace proplib
{
	// ----------------------------------------------------------------------
	// ----------------------------------------------------------------------
	// --- CLASS SchemaDocument
	// ----------------------------------------------------------------------
	// ----------------------------------------------------------------------
	class SchemaDocument : public Document
	{
	private:
		friend class DocumentBuilder;
		SchemaDocument( std::string name, std::string path );

	public:
		virtual ~SchemaDocument();

		void apply( Document *values );
		void makePathDefaults( Document *values, SymbolPath *symbolPath );

	private:
		void normalize( ObjectProperty &propertySchema, Property &propertyValue );
		void normalizeObject( ObjectProperty &propertySchema, ObjectProperty &propertyValue );
		void normalizeArray( ObjectProperty &propertySchema, ArrayProperty &propertyValue );

		void validate( Property &propertyValue );
		void validateScalar( Property &propertyValue );
		void validateScalar( ObjectProperty &schema, Property &value );
		void validateEnum( Property &propertyValue );
		void validateObject( Property &propertyValue );
		void validateArray( Property &propertyValue );
		void validateCommonAttribute( Property &attr, Property &value );

		void makePathDefaults( ObjectProperty &schema, __ContainerProperty &value, SymbolPath::Element *pathElement );

		bool isLegacyMode( Document *values );
		Property *createDefault( Property &schema );

		bool _legacyMode;
	};
}
