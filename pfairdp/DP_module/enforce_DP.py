from opacus import PrivacyEngine

def make_private(model, optimizer, train_data_loader, dp_params):
    privacy_engine = PrivacyEngine(accountant = 'gdp')
    if not bool(dp_params):
      print('\n*** DP disabeld ***\n')
    else:
      print('Instantiating the privacy engine \n')
      if not "target_epsilon" in dp_params:
        # Main execution mode -- Privacy budget depending on the value of the two hyperparameters
        model, optimizer, train_data_loader = privacy_engine.make_private(
            module = model,
            optimizer = optimizer,
            data_loader = train_data_loader,
            noise_multiplier = dp_params['noise_multiplier'],
            max_grad_norm = dp_params['max_grad_norm']
        )
      else:
        # Run pipeline with predefined privacy constraints
        eps = dp_params['target_epsilon']
        delta = dp_params['target_delta']

        print(f'***** Privacy Engine instantiated for specific privacy budget eps = {eps} and delta = {delta} *****')
        model, optimizer, train_data_loader = privacy_engine.make_private_with_epsilon(
            module = model,
            optimizer = optimizer,
            data_loader = train_data_loader,
            target_epsilon = dp_params['target_epsilon'],
            target_delta = dp_params['target_delta'],
            epochs = dp_params['epochs'],
            max_grad_norm = dp_params['max_grad_norm'],
            noise_multiplier = dp_params['noise_multiplier']
        )

    # The privacy engine allows measuring the final privacy level. 
    # We return it here because this needs to be measured after the model has been trained.
    return model, optimizer, train_data_loader, privacy_engine