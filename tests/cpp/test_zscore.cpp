#include <gtest/gtest.h>
#include <fstream>
#include <runtime/zscore_constants.hpp>
#include <third_party/json.hpp>

using json = nlohmann::json;

// --- fixture path ---
// tests run from build dir, path is relative to repo root
static const std::string CONSTANTS_PATH = NDR_TEST_ARTIFACTS_DIR "/constants.json";
static const std::string FIXTURE_PATH   = NDR_TEST_FIXTURES_DIR  "/zscore_fixture.json";

// --- load tests ---

TEST(ZScoreConstantsTest, LoadsWithoutThrowing) {
    EXPECT_NO_THROW(ndr::ZScoreConstants{CONSTANTS_PATH});
}

TEST(ZScoreConstantsTest, CorrectBlockCount) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    // constants.json has 23 blocks (1-24, missing one)
    EXPECT_EQ(zs.num_blocks(), 23);
}

TEST(ZScoreConstantsTest, CorrectChannelCount) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    EXPECT_EQ(zs.num_channels(), 256);
}

TEST(ZScoreConstantsTest, KnownBlocksExist) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    EXPECT_TRUE(zs.has_block("1"));
    EXPECT_TRUE(zs.has_block("14"));
    EXPECT_FALSE(zs.has_block("999"));
}

// --- Invariant tests ---

TEST(ZScoreConstantsTest, NormalizeBeforeSetBlockThrows) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    auto input = torch::zeros({1, 256});
    EXPECT_THROW(zs.normalize(input), std::runtime_error);
}

TEST(ZScoreConstantsTest, SetBlockUnkonwnIdThrows) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    EXPECT_THROW(zs.set_block("999"), std::runtime_error);
}

TEST(ZScoreConstantsTest, NormalizeWrongDtypeThrows) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    zs.set_block("1");
    auto input = torch::zeros({1, 256}, torch::kFloat64);
    EXPECT_THROW(zs.normalize(input), std::runtime_error);
}

TEST(ZScoreConstantsTest, NormalizeWrongChannelDimThrow) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    zs.set_block("1");
    auto input = torch::zeros({1, 128});
    EXPECT_THROW(zs.normalize(input), std::runtime_error);
}

TEST(ZScoreConstantsTest, NormalizeWrongNDimThrows) {
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    zs.set_block("1");
    auto input = torch::zeros({256}); // 1D, not 2D
    EXPECT_THROW(zs.normalize(input), std::runtime_error);
}

// --- correctness test ---

TEST(ZScoreConstantsTest, NormalizeMatchesPython) {
    // load fixture written by python's blockZScorer
    std::fstream f(FIXTURE_PATH);
    ASSERT_TRUE(f.is_open()) << "Fixture not found at: " << FIXTURE_PATH; 
    json fixture = json::parse(f);

    std::string block_id            = fixture.at("block_id").get<std::string>();
    std::vector<float> input_vec    = fixture.at("input")[0].get<std::vector<float>>();
    std::vector<float> expected_vec = fixture.at("expected")[0].get<std::vector<float>>();

    // build [1,256] input tensor from fixture
    auto input    = torch::tensor(input_vec).unsqueeze(0); // [1, 256]
    auto expected = torch::tensor(expected_vec).unsqueeze(0); // [1, 256]
    
    ndr::ZScoreConstants zs(CONSTANTS_PATH);
    zs.set_block(block_id);
    auto output = zs.normalize(input);

    ASSERT_TRUE(torch::allclose(output, expected, /*atol=*/1e-5, /*rtol=*/1e05))
        << "Max absolute difference: "
        << (output - expected).abs().max().item<float>();
}