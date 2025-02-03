using System;
using System.Linq;
using Codesigner;
using Codesigner.Models;
using Newtonsoft.Json;
using Xunit;
using Xunit.Abstractions;

namespace CodesignerTest
{
    public class CodesignerUnitTests : IClassFixture<TestFixture>
    {
        private readonly GrasshopperAIClient _client;
        private readonly ITestOutputHelper _output;

        public CodesignerUnitTests(TestFixture fixture, ITestOutputHelper output)
        {
            this._client = fixture.Client;
            this._output = output;
        }

        [Fact]
        public void CreatesRectangleWithNumberSliders()
        {
            var prompt = "Create a rectangle and number sliders to control it.";

            // Act: Generate the canvas from the prompt
            var response = _client.GenerateGrasshopper<GrasshopperSchema>(prompt);
            _output.WriteLine(JsonConvert.SerializeObject(response));
            // Assert: Validate the response is not null and contains expected components
            Assert.NotNull(response);

            // Check that the rectangle component exists
            var rectangleComponent = response.Components
                .FirstOrDefault(component => component.Name.Equals("Rectangle", StringComparison.OrdinalIgnoreCase));
            Assert.NotNull(rectangleComponent);


        }

    }
}

