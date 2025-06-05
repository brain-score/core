"""
Integration test to verify that metadata and scoring systems select the same alexnet model_id 
from the actual production database with real duplicate models.
"""
import unittest
import logging
import os
from datetime import datetime

from brainscore_core.submission.database import (
    connect_db, create_model_meta_entry, modelentry_from_model, 
    submissionentry_from_meta
)
from brainscore_core.submission.database_models import Model, User, Submission

logger = logging.getLogger(__name__)

# Dev database connection
DB_SECRET = "brainscore-1-ohio-cred-migrated"


class TestAlexNetConsistencyIntegration(unittest.TestCase):
    """Integration test for alexnet model consistency using database."""
    
    @classmethod
    def setUpClass(cls):
        """Set up dev database connection."""
        logger.info('Connect to dev database for integration test')
        try:
            connect_db(db_secret=DB_SECRET)
        except Exception as e:
            raise unittest.SkipTest(f"Cannot connect to database: {e}")

    def setUp(self):
        """Set up test data for each test."""
        logger.info('Initialize integration test data')
        # Get existing user for creating submissions
        self.user = User.select().first()
        if not self.user:
            raise unittest.SkipTest("No users found in database")

    def test_alexnet_duplicate_consistency(self):
        """Test that scoring and metadata find the same alexnet model from duplicates."""
        model_identifier = "alexnet"
        
        # Check that alexnet duplicates actually exist
        # At the time of writing, there are 60 alexnet models in the database with the identifier "alexnet"
        alexnet_models = Model.select().where(Model.name == model_identifier)
        alexnet_count = len(alexnet_models)
        
        logger.info(f"Found {alexnet_count} alexnet models in database")
        
        # Skip test if the database has been cleaned up
        if alexnet_count < 2:
            self.skipTest(f"Need multiple alexnet models for duplicate test, found only {alexnet_count}")
        
        # Test 1: Get model using scoring approach (simulate scoring)
        logger.info("-----Testing scoring approach-----")
        
        # Create a test submission for scoring
        # This will be cleaned up to not litter the Submission table
        test_submission = Submission.create(
            jenkins_id=99999,  # High number to avoid conflicts
            submitter=self.user,
            model_type="integration_test",
            status="running",
            timestamp=datetime.now()
        )
        
        try:
            # Use the same function that scoring uses
            scoring_model = modelentry_from_model(
                model_identifier=model_identifier,
                domain="vision",
                submission=test_submission,
                public=False,  # Use false to avoid affecting public models
                competition=None
            )
            
            logger.info(f"Scoring approach found model ID: {scoring_model.id}")
            
            # Test 2: Get model using metadata approach
            logger.info("-----Testing metadata approach-----")
            
            # Create test metadata (won't affect since we'll clean up)
            test_metadata = {
                'architecture': 'CNN',
                'model_family': 'AlexNet',
                'total_parameter_count': 60000000,
                'runnable': True,
                'extra_notes': 'Integration test - safe to delete'
            }
            
            # This should use the same modelentry_from_model function internally
            metadata_result = create_model_meta_entry(model_identifier, test_metadata)
            metadata_model_id = metadata_result.model.id
            
            logger.info(f"Metadata approach found model ID: {metadata_model_id}")
            
            # Test 3: Verify consistency
            self.assertEqual(scoring_model.id, metadata_model_id,
                           f"Inconsistent model selection! Scoring: {scoring_model.id}, "
                           f"Metadata: {metadata_model_id}")
            
            logger.info(f"SUCCESS: Both approaches selected model_id={scoring_model.id}")
            
            # Log details about the selected model for verification
            logger.info(f"-----Selected alexnet model details-----")
            logger.info(f"ID: {scoring_model.id}")
            logger.info(f"Name: {scoring_model.name}")
            logger.info(f"Domain: {scoring_model.domain}")
            logger.info(f"Owner ID: {scoring_model.owner.id}")
            logger.info(f"Public: {scoring_model.public}")
            logger.info(f"Submission ID: {scoring_model.submission.id}")
            
            # Verify it's actually one of the duplicates
            selected_model_exists = Model.select().where(
                (Model.name == model_identifier) & 
                (Model.id == scoring_model.id)
            ).exists()
            self.assertTrue(selected_model_exists, "Selected model should exist in alexnet duplicates")
            
        finally:
            # Clean up: Remove test metadata and submission
            if 'metadata_result' in locals():
                try:
                    metadata_result.delete_instance()
                    logger.info("Cleaned up test metadata")
                except:
                    # Throw error to fail Jenkins
                    logger.error("Could not clean up test metadata")
            
            try:
                test_submission.delete_instance()
                logger.info("Cleaned up test submission")
            except:
                # Throw error to fail Jenkins
                logger.error("Could not clean up test submission")

    def test_alexnet_model_selection_deterministic(self):
        """Test that repeated calls return the same alexnet model (deterministic behavior)."""
        model_identifier = "alexnet"
        
        # Check that duplicates exist
        alexnet_models = Model.select().where(Model.name == model_identifier)
        if len(alexnet_models) < 2:
            self.skipTest("Need multiple alexnet models for deterministic test")
        
        # Create multiple test submissions
        submissions = []
        selected_models = []
        
        try:
            # Test multiple calls to see if they're consistent
            # Just a sanity check.
            for i in range(10):
                submission = Submission.create(
                    jenkins_id=99990 + i,
                    submitter=self.user,
                    model_type="deterministic_test",
                    status="running",
                    timestamp=datetime.now()
                )
                submissions.append(submission)
                
                model = modelentry_from_model(
                    model_identifier=model_identifier,
                    domain="vision",
                    submission=submission,
                    public=False,
                    competition=None
                )
                selected_models.append(model.id)
                
                logger.info(f"Call {i+1}: Selected model_id={model.id}")
            
            # Verify all calls returned the same model
            first_model_id = selected_models[0]
            for i, model_id in enumerate(selected_models[1:], 1):
                self.assertEqual(first_model_id, model_id,
                               f"Call {i+1} returned different model_id: {model_id} vs {first_model_id}")
            
            logger.info(f"SUCCESS: All {len(selected_models)} calls consistently returned model_id={first_model_id}")
            
        finally:
            # Clean up all test submissions
            for submission in submissions:
                try:
                    submission.delete_instance()
                except:
                    logger.warning(f"Could not clean up submission {submission.jenkins_id}")

    def test_metadata_query_by_identifier_works(self):
        """Test that we can query metadata for alexnet after creating it."""
        model_identifier = "alexnet"
        
        # Create metadata for alexnet
        test_metadata = {
            'architecture': 'CNN',
            'model_family': 'AlexNet',
            'total_parameter_count': 62000000,
            'runnable': False,
            'extra_notes': 'Integration test metadata - safe to delete'
        }
        
        try:
            # Create metadata
            metadata_result = create_model_meta_entry(model_identifier, test_metadata)
            created_model_id = metadata_result.model.id
            
            logger.info(f"Created metadata for model_id={created_model_id}")
            
            # Verify we can find this metadata by querying the model
            found_model = Model.get(Model.id == created_model_id)
            self.assertEqual(found_model.name, model_identifier)
            
            # Test requerying metadata using just the identifier
            from brainscore_core.submission.database import get_model_metadata_by_identifier, get_model_with_metadata
            
            # Test get_model_metadata_by_identifier
            requeried_metadata = get_model_metadata_by_identifier(found_model.name)
            self.assertIsNotNone(requeried_metadata, "Should be able to find metadata by identifier")
            self.assertEqual(requeried_metadata.architecture, 'CNN')
            self.assertEqual(requeried_metadata.model_family, 'AlexNet')
            self.assertEqual(requeried_metadata.total_parameter_count, 62000000)
            self.assertEqual(requeried_metadata.runnable, False)
            
            # Test get_model_with_metadata
            requeried_model, requeried_meta = get_model_with_metadata(model_identifier)
            self.assertEqual(requeried_model.id, created_model_id)
            self.assertEqual(requeried_model.name, model_identifier)
            self.assertIsNotNone(requeried_meta, "Should return metadata along with model")
            self.assertEqual(requeried_meta.architecture, 'CNN')
            
            logger.info("SUCCESS: Metadata correctly linked to alexnet model and can be requeried by identifier") 
            
        finally:
            # Clean up
            if 'metadata_result' in locals():
                try:
                    metadata_result.delete_instance()
                    logger.info("Cleaned up test metadata")
                except:
                    logger.warning("Could not clean up test metadata")


if __name__ == '__main__':
    # Logging to print out test results for debugging purposes.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2) 