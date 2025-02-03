using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Newtonsoft.Json;
using Newtonsoft.Json.Schema;
using Newtonsoft.Json.Schema.Generation;
using OpenAI.Chat;

namespace Codesigner.Structured
{
    internal static class StructuredOutputsExtensions
    {

        public static ChatResponseFormat CreateJsonSchemaFormat<T>(string jsonSchemaFormatName, string jsonSchemaFormatDescription = null, bool jsonSchemaIsStrict = false)
        {
            string jsonSchema = GetJsonSchema(typeof(T));

            Console.WriteLine(jsonSchema);

            var bytes = Encoding.UTF8.GetBytes(jsonSchema);

            return ChatResponseFormat.CreateJsonSchemaFormat(
                jsonSchemaFormatName,
                jsonSchema: BinaryData.FromBytes(bytes),
                jsonSchemaFormatDescription: jsonSchemaFormatDescription,
                jsonSchemaIsStrict: jsonSchemaIsStrict
            );
        }

        private static string GetJsonSchema(Type type)
        {
            JSchemaGenerator generator = new JSchemaGenerator
            {
                DefaultRequired = Required.Always, // Ensures required fields are set
                SchemaReferenceHandling = SchemaReferenceHandling.None // Prevents `$ref`
            };

            JSchema schema = generator.Generate(type);

            SetAdditionalPropertiesFalse(schema);
            // Set top-level metadata.
            schema.Title = "GrasshopperCanvas";
            schema.ExtensionData["$schema"] = "http://json-schema.org/draft-07/schema#";

            // Recursively enforce additionalProperties:false for all object types.
            SetAdditionalPropertiesFalse(schema);
            foreach (var def in schema.Properties.Values)
            {
                SetAdditionalPropertiesFalse(def);
            }

            return schema.ToString();
        }

        /// <summary>
        /// Recursively sets additionalProperties to false for every schema with type "object".
        /// </summary>
        private static void SetAdditionalPropertiesFalse(JSchema schema)
        {
            if (schema == null)
                return;

            // If the schema accepts an object, disable additional properties.
            if ((schema.Type & JSchemaType.Object) == JSchemaType.Object)
            {
                schema.AllowAdditionalProperties = false;
            }

            // Process properties of this schema.
            foreach (var prop in schema.Properties.Values)
            {
                SetAdditionalPropertiesFalse(prop);
            }

            // Process items if the schema is an array.
            if (schema.Items != null)
            {
                foreach (var item in schema.Items)
                {
                    SetAdditionalPropertiesFalse(item);
                }
            }

            // Process oneOf, anyOf, and allOf collections.
            if (schema.OneOf != null)
            {
                foreach (var s in schema.OneOf)
                {
                    SetAdditionalPropertiesFalse(s);
                }
            }
            if (schema.AnyOf != null)
            {
                foreach (var s in schema.AnyOf)
                {
                    SetAdditionalPropertiesFalse(s);
                }
            }
            if (schema.AllOf != null)
            {
                foreach (var s in schema.AllOf)
                {
                    SetAdditionalPropertiesFalse(s);
                }
            }
        }
    }
}
