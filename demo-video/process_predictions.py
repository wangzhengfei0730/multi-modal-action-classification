with open('prediction.txt', 'r') as fpr:
    lines = fpr.readlines()

with open('result.txt', 'w') as fpw:
    frame_cnt = 0
    i = 1
    while i < len(lines):
        fpw.write('frame #{0}\n'.format(frame_cnt))

        label = int(lines[i + 1].split(',')[0].split('=')[-1].strip())
        winner = int(lines[i + 1].split(',')[-1].split('=')[-1].strip())
        fpw.write('label = {0}, fusion = {1}\n'.format(label, winner))

        predictions = lines[i + 2].split('=')[-1].strip()[1:-1].split(', ')
        modals = ['skeleton', 'rgb', 'optical-flow', 'depth', 'infrared', 'infrared-depth']
        each_modal = []
        for j in range(6):
            each_modal.append(modals[j])
            each_modal.append(': ')
            each_modal.append(str(predictions[j]))
            each_modal.append('; ')
        fpw.write('{0}\n'.format(''.join(each_modal)))
            
        i += 3
        frame_cnt += 1
