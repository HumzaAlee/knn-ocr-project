import os

def main():
    while True:
        print('\n=== OCR Main Menu ===')
        print('1. Train Model')
        print('2. Test Model')
        print('3. Exit')
        
        choice = input('Enter your choice (1/2/3): ').strip()
        
        if choice == '1':
            print('\nTraining Mode:')
            os.system('python train.py')
        
        elif choice == '2':
            print('\nTesting Mode:')
            os.system('python test.py')
        
        elif choice == '3':
            print('\nExiting.')
            break
        
        else:
            print('Invalid choice. Please try again.')

if __name__ == '__main__':
    main()
